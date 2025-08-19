
import math

import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F


from src.utils import get_inception_model, get_features, calculate_fid


# =============================
# Sinusoidal Positional Embedding
# =============================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# =============================
# Residual Block
# =============================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, use_attention=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.use_attention = use_attention
        if use_attention:
            self.attn = nn.MultiheadAttention(out_channels, num_heads=4, batch_first=True)

    def forward(self, x, t):
        h = self.block1(x)
        time_emb = self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)
        h = h + time_emb
        h = self.block2(h)
        h = h + self.res_conv(x)

        if self.use_attention:
            b, c, hgt, wdt = h.shape
            h_reshape = h.view(b, c, hgt * wdt).transpose(1, 2)
            attn_out, _ = self.attn(h_reshape, h_reshape, h_reshape)
            h = attn_out.transpose(1, 2).view(b, c, hgt, wdt) + h
        return h


# =============================
# Down / Up
# =============================
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, use_attention=False):
        super().__init__()
        self.block = ResidualBlock(in_channels, out_channels, time_emb_dim, use_attention)
        self.down = nn.Conv2d(out_channels, out_channels, 4, 2, 1)

    def forward(self, x, t):
        x = self.block(x, t)
        skip = x
        x = self.down(x)
        return x, skip


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, use_attention=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)
        self.block = ResidualBlock(out_channels * 2, out_channels, time_emb_dim, use_attention)

    def forward(self, x, skip, t):
        x = self.up(x)
        # 对齐尺寸
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        x = self.block(x, t)
        return x


# =============================
# UNet
# =============================
class UNet(nn.Module):
    def __init__(self, cfg, channels=3, base_channels=64, num_classes = None):
        super().__init__()
        time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.label_emb = None
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

        self.in_conv = nn.Conv2d(channels, base_channels, 3, padding=1)

        self.down1 = Down(base_channels, base_channels * 2, time_dim)
        self.down2 = Down(base_channels * 2, base_channels * 4, time_dim, use_attention=True)
        self.down3 = Down(base_channels * 4, base_channels * 8, time_dim)

        self.bot1 = ResidualBlock(base_channels * 8, base_channels * 16, time_dim, use_attention=True)
        self.bot2 = ResidualBlock(base_channels * 16, base_channels * 16, time_dim)

        self.up1 = Up(base_channels * 16, base_channels * 8, time_dim)
        self.up2 = Up(base_channels * 8, base_channels * 4, time_dim, use_attention=True)
        self.up3 = Up(base_channels * 4, base_channels * 2, time_dim)

        self.out_conv = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, channels, 1)
        )

        self.cfg = cfg

    def forward(self, x, t, y=None):
        t = self.time_mlp(t)
        if self.label_emb is not None:
            if y is not None:
                t = t + self.label_emb(y)
            else:
                t = t + torch.zeros_like(t)

        h = self.in_conv(x)
        if self.cfg.debug_shapes:
            print("[DEBUG] in_conv", h.shape)

        h1, s1 = self.down1(h, t)
        if self.cfg.debug_shapes:
            print("[DEBUG] down1", h1.shape, "skip", s1.shape)

        h2, s2 = self.down2(h1, t)
        if self.cfg.debug_shapes:
            print("[DEBUG] down2", h2.shape, "skip", s2.shape)

        h3, s3 = self.down3(h2, t)
        if self.cfg.debug_shapes:
            print("[DEBUG] down3", h3.shape, "skip", s3.shape)

        h = self.bot1(h3, t)
        if self.cfg.debug_shapes:
            print("[DEBUG] bottleneck1", h.shape)
        h = self.bot2(h, t)
        if self.cfg.debug_shapes:
            print("[DEBUG] bottleneck2", h.shape)

        h = self.up1(h, s3, t)
        if self.cfg.debug_shapes:
            print("[DEBUG] up1", h.shape)
        h = self.up2(h, s2, t)
        if self.cfg.debug_shapes:
            print("[DEBUG] up2", h.shape)
        h = self.up3(h, s1, t)
        if self.cfg.debug_shapes:
            print("[DEBUG] up3", h.shape)

        # h = torch.cat([h, s1], dim=1)
        out = self.out_conv(h)
        if self.cfg.debug_shapes:
            print("[DEBUG] out_conv", out.shape)
        return out


# =============================
# Noise Schedule
# =============================
def linear_beta_schedule(timesteps):
    beta_start = 1e-4
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)



def q_sample(x_start, t, noise,alphas_cumprod):
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[t])
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod[t])
    return sqrt_alphas_cumprod[:, None, None, None] * x_start + sqrt_one_minus_alphas_cumprod[:, None, None, None] * noise


# =============================
# DDIM Sampling
# =============================
def p_sample_step(model, x_t, t, y, guidance_scale=3.0,T=1000):
    t_norm = t.float() / T

    # 有条件预测
    pred_cond = model(x_t, t_norm, y)
    # 无条件预测
    pred_uncond = model(x_t, t_norm, None)

    # CFG 组合
    pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
    return pred


@torch.no_grad()
def sample_ddim_cfg(model, label,Config,device,alphas,alphas_cumprod,n_samples=8, sample_steps=50, eta=0.0, guidance_scale=3.0):
    model.eval()
    shape = (n_samples, 3, Config.image_size, Config.image_size)
    img = torch.randn(shape, device=device)

    step_size = Config.T // sample_steps
    times = list(range(0, Config.T, step_size))
    times_next = [-1] + times[:-1]

    for i, j in zip(reversed(times), reversed(times_next)):
        t = torch.tensor([i] * n_samples, device=Config.device)
        t_norm = t.float() / Config.T
        # pred_noise = model(img, t_norm)
        pred_noise = p_sample_step(model, img,t, torch.tensor([label] * n_samples).to(Config.device),guidance_scale=guidance_scale)
        alpha = alphas[i]
        alpha_cum = alphas_cumprod[i]
        pred_x0 = (img - torch.sqrt(1 - alpha_cum) * pred_noise) / torch.sqrt(alpha_cum)
        if j < 0:
            img = pred_x0
            continue

        alpha_cum_next = alphas_cumprod[j]
        sigma = (
            eta
            * torch.sqrt((1 - alpha_cum_next) / (1 - alpha_cum))
            * torch.sqrt(1 - alpha_cum / alpha_cum_next)
        )
        noise = torch.randn_like(img) if sigma > 0 else 0

        img = torch.sqrt(alpha_cum_next) * pred_x0 + torch.sqrt(1 - alpha_cum_next - sigma**2) * pred_noise + sigma * noise

    img = (img.clamp(-1, 1) + 1) / 2
    return img






# 主入口：计算真实 vs 生成图像的 FID
def compute_fid(model, dataloader_real, device,Config, n_samples=5000, sample_steps=50):
    inception = get_inception_model(device)

    # 真实图像特征
    real_features = get_features(dataloader_real, inception, device, max_batches=n_samples // dataloader_real.batch_size)
    mu_real, sigma_real = real_features.mean(axis=0), np.cov(real_features, rowvar=False)

    # 生成图像特征
    model.eval()
    fake_images = []
    with torch.no_grad():
        for _ in range(n_samples // dataloader_real.batch_size):
            z = torch.randn(dataloader_real.batch_size, 3, Config.image_size, Config.image_size, device=device)
            t = torch.full((dataloader_real.batch_size,), Config.T-1, device=device)
            t_norm = t.float() / Config.T
            fake = model(z, t_norm).clamp(-1, 1)
            fake = (fake + 1) / 2
            fake_images.append(fake)
    fake_images = torch.cat(fake_images, dim=0)

    fake_features = []
    for i in range(0, len(fake_images), dataloader_real.batch_size):
        batch = fake_images[i:i+dataloader_real.batch_size]
        feat = inception(batch)[0] if isinstance(inception(batch), tuple) else inception(batch)
        fake_features.append(feat.detach().cpu().numpy())
    fake_features = np.concatenate(fake_features, axis=0)

    mu_fake, sigma_fake = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)

    fid = calculate_fid(mu_real, sigma_real, mu_fake, sigma_fake)
    return fid




