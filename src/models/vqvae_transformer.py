import os

import torch
from torch import nn
import torch.nn.functional as F
import math
from pathlib import Path
from torchvision import transforms, datasets, utils as vutils
from tqdm import tqdm


def exists(x):
    return x is not None

def save_image_grid(tensor, path, nrow=4, normalize=True, value_range=(-1, 1)):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(tensor, str(path), nrow=nrow, normalize=normalize, value_range=value_range)
# ----------------------------
# VQ-VAE Components
# ----------------------------


class Encoder(nn.Module):
    def __init__(self, in_ch=3, D=256):
        super().__init__()
        ch = 128
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, ch, 4, 2, 1), nn.ReLU(True),       # /2
            nn.Conv2d(ch, ch, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(ch, ch*2, 4, 2, 1), nn.ReLU(True),        # /4
            nn.Conv2d(ch*2, ch*2, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(ch*2, ch*4, 4, 2, 1), nn.ReLU(True),      # /8
            nn.Conv2d(ch*4, D, 4, 2, 1),                        # /16
        )
    def forward(self, x): return self.net(x)

class Decoder(nn.Module):
    def __init__(self, D=256, out_ch=3):
        super().__init__()
        ch = 128
        self.net = nn.Sequential(
            nn.ConvTranspose2d(D, ch*4, 4, 2, 1), nn.ReLU(True),    # x2
            nn.ConvTranspose2d(ch*4, ch*2, 4, 2, 1), nn.ReLU(True), # x4
            nn.ConvTranspose2d(ch*2, ch*2, 4, 2, 1), nn.ReLU(True), # x8
            nn.ConvTranspose2d(ch*2, ch,   4, 2, 1), nn.ReLU(True), # x16
            nn.Conv2d(ch, out_ch, 3, 1, 1),
            nn.Tanh(),  # output in [-1,1]
        )
    def forward(self, z): return self.net(z)

class VectorQuantizerEMA(nn.Module):
    """EMA VQ as in VQ-VAE-2; more stable than plain MSE codebook updates."""
    def __init__(self, n_embed=2048, embed_dim=256, beta=0.25, decay=0.99, eps=1e-5):
        super().__init__()
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.beta = beta
        self.decay = decay
        self.eps = eps

        embed = torch.randn(embed_dim, n_embed)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, z_e):
        # z_e: (B, D, H, W)
        B, D, H, W = z_e.shape
        ze = z_e.permute(0,2,3,1).contiguous().view(-1, D)   # (BHW, D)

        # distances: (BHW, n_embed) = ||ze||^2 - 2*ze*e + ||e||^2
        dist = (ze.pow(2).sum(1, keepdim=True)
                - 2 * ze @ self.embed
                + self.embed.pow(2).sum(0, keepdim=True))
        idx = dist.argmin(dim=1)                              # (BHW,)
        z_q = self.embed[:, idx].T.view(B, H, W, D).permute(0,3,1,2).contiguous()

        # EMA updates (training only)
        if self.training:
            with torch.no_grad():
                one_hot = F.one_hot(idx, self.n_embed).type_as(ze)  # (BHW, n_embed)
                cluster_size = one_hot.sum(0)                       # (n_embed,)
                self.cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)

                embed_sum = ze.T @ one_hot                          # (D, n_embed)
                self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

                n = self.cluster_size.sum()
                cluster_size = (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
                embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
                self.embed.copy_(embed_normalized)

        # losses
        commit_loss = self.beta * F.mse_loss(z_e.detach(), z_q)
        # straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()
        return z_q_st, idx.view(B, H, W), commit_loss

    def decode_from_indices(self, idx_map):
        # idx_map: (B, H, W)
        B, H, W = idx_map.shape
        z = self.embed[:, idx_map.view(-1)].T.view(B, H, W, self.embed_dim).permute(0,3,1,2).contiguous()
        return z

class VQVAE(nn.Module):
    def __init__(self, codebook_size=2048, D=256):
        super().__init__()
        self.enc = Encoder(D=D)
        self.vq  = VectorQuantizerEMA(n_embed=codebook_size, embed_dim=D)
        self.dec = Decoder(D=D)
    def forward(self, x):
        z_e = self.enc(x)
        z_q, idx, vq_loss = self.vq(z_e)
        x_rec = self.dec(z_q)
        rec_l1 = F.l1_loss(x_rec, x)
        loss = rec_l1 + vq_loss
        return x_rec, idx, loss
    def encode_indices(self, x):
        with torch.no_grad():
            z_e = self.enc(x)
            _, idx, _ = self.vq(z_e)
        return idx  # (B, H, W)
    def decode_indices(self, idx):
        with torch.no_grad():
            z_q = self.vq.decode_from_indices(idx)
            x_rec = self.dec(z_q)
        return x_rec

def train_vqvae(model, train_dataloader,optim_vqvae,save_dir, par, device):
    best_loss = float('inf')
    for epoch in range(par.epochs):
        model.train()
        pbar = tqdm(train_dataloader, desc=f'[VQ-VAE] Epoch {epoch + 1}/{par.epochs}')
        running = 0.0
        for x, _ in pbar:
            # print(x.max(), x.min())
            x = x.to(device, non_blocking=True)
            # with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            x_rec, _, loss = model(x)
            optim_vqvae.zero_grad(set_to_none=True)
            loss.backward()
            # scaler.step(opt)
            # scaler.update()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optim_vqvae.step()
            running += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        avg = running / len(train_dataloader)
        # sample a small grid
        model.eval()
        with torch.no_grad():
            x = next(iter(train_dataloader))[0][:16].to(device)
            x_rec, _, _ = model(x)
            save_image_grid(x, os.path.join(save_dir , f'samples_vqvae_epoch{epoch + 1}_real.png'), nrow=4)
            save_image_grid(x_rec, os.path.join(save_dir , f'samples_vqvae_epoch{epoch + 1}_rec.png'), nrow=4)
        # save
        ckpt_path = os.path.join(save_dir , f'vqvae_{epoch}.pt')
        torch.save(model.state_dict(), ckpt_path)
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), os.path.join(save_dir , f'vqvae_best.pt'))
        print(f'Epoch {epoch + 1} avg loss: {avg:.4f}')



# ----------------------------
# GPT-like Transformer for tokens
# ----------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model=512, n_head=8, dropout=0.0, seq_len=256):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.key = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(dropout)
        # causal mask (1 for keep, 0 for block)
        self.register_buffer("mask", torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C//self.n_head).transpose(1,2)   # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C//self.n_head).transpose(1,2)
        v = self.value(x).view(B, T, self.n_head, C//self.n_head).transpose(1,2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        mask = self.mask[:, :, :T, :T]
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1,2).contiguous().view(B, T, C)
        y = self.proj_drop(self.proj(y))
        return y

class Block(nn.Module):
    def __init__(self, d_model, n_head, dropout, seq_len):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout, seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TokenTransformer(nn.Module):
    def __init__(self, vocab_size, seq_len=16*16, d_model=512, n_head=8, n_layer=12, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d_model, n_head, dropout, seq_len) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and exists(m.bias):
            nn.init.zeros_(m.bias)

    def forward(self, idx):
        # idx: (B, T)
        B, T = idx.shape
        x = self.tok(idx) + self.pos[:, :T, :]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab)
        return logits

    @torch.no_grad()
    def generate(self, idx_start, steps, temperature=1.0, top_k=64):
        # idx_start: (B, T0) start tokens
        self.eval()
        idx = idx_start
        for _ in range(steps):
            idx_cond = idx[:, -self.seq_len:]
            logits = self(idx_cond)[:, -1, :] / max(temperature, 1e-6)  # last step
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx


def _flatten_indices(idx_map):
    # idx_map: (B, H, W) -> (B, H*W)
    return idx_map.view(idx_map.size(0), -1)

def _shift_targets(x):
    # x: (B, T) -> inputs (B, T-1), targets (B, T-1)
    return x[:, :-1].contiguous(), x[:, 1:].contiguous()

def train_transformer(vqvae, gpt, train_dataloader, optim_gpt, save_dir, par, device):
    best_loss = float('inf')
    for epoch in range(par.epochs):
        gpt.train()
        pbar = tqdm(train_dataloader, desc=f'[GPT] Epoch {epoch + 1}/{par.epochs}')
        running = 0.0
        for x, _ in pbar:
            x = x.to(device, non_blocking=True)
            with torch.no_grad():
                idx_map = vqvae.encode_indices(x)  # (B, H, W)
                seq = _flatten_indices(idx_map)  # (B, T)
            # teacher forcing: shift
            inp, tgt = _shift_targets(seq)

            logits = gpt(inp)  # (B, T-1, vocab)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            optim_gpt.zero_grad(set_to_none=True)
            loss.backward()
            optim_gpt.step()

            running += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4f}')
        avg = running / len(train_dataloader)
        torch.save(gpt.state_dict(), os.path.join(save_dir , f'gpt_{epoch}.pt'))
        if avg < best_loss:
            best_loss = avg
            torch.save(gpt.state_dict(), os.path.join(save_dir , 'gpt_best.pt'))
        print(f'Epoch {epoch + 1} avg NLL: {avg:.4f}')


def sample_images(vqvae_ckpt,gpt_ckpt,save_dir,is_best=False):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    import yaml
    from experiments.configs.par import Struct
    par = yaml.safe_load(open(os.path.join(save_dir,'Generation.yaml'), 'r'))
    par = Struct(**par)
    if is_best:
        vq_state = torch.load(os.path.join(save_dir,'vqvae_best.pt'), map_location='cpu')
        gpt_state = torch.load(os.path.join(save_dir, 'gpt_best.pt'), map_location='cpu')
    else:

        vq_state = torch.load(vqvae_ckpt, map_location='cpu')
        gpt_state = torch.load(gpt_ckpt, map_location='cpu')



    D = par.D
    img_size = par.img_size
    H = W = img_size // 16
    T = H * W

    vqvae = VQVAE(codebook_size=par.codebook_size, D=D).to(device)
    vqvae.load_state_dict(vq_state)
    vqvae.eval()



    gpt = TokenTransformer(vocab_size=par.codebook_size, seq_len=par.seq_len,
                           d_model=par.gpt_d_model,
                           n_head=par.gpt_heads,
                           n_layer=par.gpt_layers,
                           dropout=par.gpt_dropout).to(device)
    gpt.load_state_dict(gpt_state)
    gpt.eval()

    B = par.num_samples
    # start with a BOS (here we just use zeros as a simple BOS), then autoregress T-1 steps
    start = torch.zeros(B, 1, dtype=torch.long, device=device)
    with torch.no_grad():
        seq = gpt.generate(start, steps=T-1, temperature=par.temperature, top_k=par.top_k)  # (B, T)
        seq = seq[:, 1:]  # drop the first dummy token to make length T-1 -> T-? Align to T:
        # If length < T, pad; if >T, trim
        if seq.size(1) < T:
            pad = torch.zeros(B, T - seq.size(1), dtype=torch.long, device=device)
            seq = torch.cat([seq, pad], dim=1)
        elif seq.size(1) > T:
            seq = seq[:, :T]
        idx_map = seq.view(B, H, W)
        imgs = vqvae.decode_indices(idx_map)

    save_image_grid(imgs, os.path.join(save_dir , f'samples_gen_{B}.png'), nrow=int(math.sqrt(B) + 0.5))
    print(f"Saved {B} generated images to {save_dir}")