import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm


class UNet224(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, time_emb_dim=256):
        super().__init__()

        # 时间步嵌入（更高维）
        self.time_embed = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # 编码器 (下采样)
        self.encoder = nn.ModuleList([
            DoubleConv(in_channels, base_channels),  # [224] -> [224]
            Downsample(base_channels, base_channels * 2,time_emb_dim),  # [224] -> [112]
            Downsample(base_channels * 2, base_channels * 4,time_emb_dim),  # [112] -> [56]
            Downsample(base_channels * 4, base_channels * 8,time_emb_dim),  # [56] -> [28]
            Downsample(base_channels * 8, base_channels * 16, time_emb_dim,use_attn=True),  # [28] -> [14] (加注意力)
            Downsample(base_channels * 16, base_channels * 16, time_emb_dim,use_attn=True)  # [14] -> [7] (加注意力)
        ])

        # 中间层 (Bottleneck)
        self.mid = nn.ModuleList([
            ResBlock(base_channels * 16, base_channels * 16, time_emb_dim),
            AttentionBlock(base_channels * 16),
            ResBlock(base_channels * 16, base_channels * 16, time_emb_dim),
        ])

        # 解码器 (上采样)
        self.decoder = nn.ModuleList([
            Upsample(base_channels * 16, base_channels * 16, time_emb_dim, use_attn=True),  # [7] -> [14]
            Upsample(base_channels * 16, base_channels * 8, time_emb_dim, use_attn=True),  # [14] -> [28] (拼接后通道数x2)
            Upsample(base_channels * 8, base_channels * 4, time_emb_dim),  # [28] -> [56]
            Upsample(base_channels * 4, base_channels * 2, time_emb_dim),  # [56] -> [112]
            Upsample(base_channels * 2, base_channels, time_emb_dim),  # [112] -> [224]
        ])

        # 输出层
        self.out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, t):
        # 时间步嵌入
        t_emb = sinusoidal_embedding(t, self.time_embed[0].in_features)
        t_emb = self.time_embed(t_emb).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

        # 编码器路径
        skip_connections = []
        for layer in self.encoder:
            if isinstance(layer, DoubleConv):
                x = layer(x)
            else:
                x = layer(x, t_emb)
            skip_connections.append(x)
            # x = layer(x, t_emb) if not isinstance(layer, DoubleConv) else layer(x)
            # skip_connections.append(x)
        skip_connections.pop()
        # 中间层
        for layer in self.mid:
            x = layer(x, t_emb)  if isinstance(layer,ResBlock) else layer(x)

        # 解码器路径（注意跳跃连接的拼接）
        for layer in self.decoder:
            if isinstance(layer, Upsample):
                x = layer(x, skip_connections.pop(),t_emb)
            else:
                x = layer(x)

        # 最终输出（融合最后一个跳跃连接）
        # x = torch.cat([x, skip_connections.pop()], dim=1)  # 拼接最初的浅层特征
        return self.out(x)


# --- 核心模块（与之前相同，但调整了通道数）---
class DoubleConv(nn.Module):
    """双卷积层（输入层）"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.conv(x)


class ResBlock(nn.Module):
    """残差块（集成时间步嵌入）"""

    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
        )
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        h = self.block(x)
        h += self.mlp(t_emb.squeeze(-1).squeeze(-1)).unsqueeze(-1).unsqueeze(-1)
        return h + self.res_conv(x)


class Downsample(nn.Module):
    """下采样块（空间减半）"""

    def __init__(self, in_channels, out_channels,time_emb_dim ,use_attn=False):
        super().__init__()
        self.conv = ResBlock(in_channels, out_channels, time_emb_dim)
        self.down = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        self.attn = AttentionBlock(out_channels) if use_attn else nn.Identity()

    def forward(self, x,t):
        x = self.conv(x,t)
        x = self.down(x)
        return self.attn(x)


class Upsample(nn.Module):
    """上采样块（空间加倍）"""

    def __init__(self, in_channels, out_channels, time_emb_dim, use_attn=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = ResBlock(out_channels * 2, out_channels, time_emb_dim)  # 拼接后通道数x2
        self.attn = AttentionBlock(out_channels) if use_attn else nn.Identity()

    def forward(self, x, skip,t):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)  # 拼接跳跃连接
        x = self.conv(x,t)
        return self.attn(x)


class AttentionBlock(nn.Module):
    """自注意力机制（空间注意力）"""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q(self.norm(x)).view(B, C, -1)  # [B, C, H*W]
        k = self.k(self.norm(x)).view(B, C, -1)  # [B, C, H*W]
        v = self.v(self.norm(x)).view(B, C, -1)  # [B, C, H*W]

        attn = torch.softmax(torch.bmm(q.permute(0, 2, 1), k) / (C ** 0.5), dim=-1)  # [B, H*W, H*W]
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)  # [B, C, H, W]
        return x + self.proj(out)


# --- 辅助函数 ---
def sinusoidal_embedding(t, dim):
    """生成正弦位置编码"""
    half_dim = dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
    emb = t.unsqueeze(-1) * emb.unsqueeze(0)
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

# 定义 DDPM
class DDPM:
    def __init__(self, model, T, beta_start, beta_end, par):
        self.model = model.cuda()
        self.T = T
        # self.device = device

        # 定义 beta_t 和 alpha_t
        self.betas = torch.linspace(beta_start, beta_end, T).cuda()
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.img_size = par.img_size
    def forward_process(self, x0, t):
        """前向扩散过程：逐步加噪"""
        noise = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
        return xt, noise

    def reverse_process(self, x, t):
        """逆扩散过程：去噪生成"""
        pred_noise = self.model(x, t)
        return pred_noise

    def sample(self, n_samples=1):
        """从纯噪声生成样本"""
        with torch.no_grad():
            x = torch.randn((n_samples, 3, self.img_size, self.img_size)).cuda()
            for t in range(self.T - 1, -1, -1):
                t_tensor = torch.tensor([t]).cuda().repeat(n_samples)
                pred_noise = self.reverse_process(x, t_tensor)
                alpha_t = self.alphas[t]
                alpha_bar_t = self.alpha_bars[t]
                beta_t = self.betas[t]
                if t > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = 0
                x = (1 / torch.sqrt(alpha_t)) * (
                        x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * pred_noise
                ) + torch.sqrt(beta_t) * noise
        return x.clamp(-1, 1)


# 训练函数
def train_ddpm(train_dataloader,save_dir,par):

    unet = UNet224(3,3,64,256)

    ddpm = DDPM(unet,par.T,par.beta_start, par.beta_end, par)

    optimizer = torch.optim.Adam(ddpm.model.parameters(), lr=1e-3)
    for epoch in range(par.epochs):
        pbar = tqdm(train_dataloader)
        for x0, _ in pbar:
            x0 = x0.cuda()
            optimizer.zero_grad()

            # 随机采样时间步 t
            t = torch.randint(0, ddpm.T, (x0.shape[0],)).cuda()

            # 前向加噪
            xt, noise = ddpm.forward_process(x0, t)

            # 预测噪声
            pred_noise = ddpm.reverse_process(xt, t)

            # 计算损失
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()
            optimizer.step()

            pbar.set_description(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        torch.save(ddpm.model.state_dict(), os.path.join(save_dir, f"DDPM_{epoch}.pth"))
        generated = ddpm.sample(16)
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            ax.imshow(np.transpose((((generated[i].clamp(-1,1)) + 1)/2).cpu().numpy(), (1, 2, 0)))
            ax.axis("off")
        plt.show()
        plt.savefig(os.path.join(save_dir, f"samples_epoch_{epoch}.png"))
        plt.close()
#
# # 数据加载
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])
# dataset = MNIST(root="./data", train=True, transform=transform, download=True)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#
# # 初始化模型和 DDPM
# model = UNet()
# ddpm = DDPM(model, T, beta_start, beta_end, device)
#
# # 训练
# train(ddpm, dataloader, epochs=10)
#
# # 生成样本
# generated = ddpm.sample(n_samples=16)
#
# # 可视化生成结果
# fig, axes = plt.subplots(4, 4, figsize=(8, 8))
# for i, ax in enumerate(axes.flat):
#     ax.imshow(generated[i].squeeze().cpu().numpy(), cmap="gray")
#     ax.axis("off")
# plt.show()
def exists(x):
    return x is not None

def train_ddpm(diffusion_model, save_dir,train_loader,par):
    channels = diffusion_model.channels
    is_ddim_sampling = diffusion_model.is_ddim_sampling

    save_and_sample_every = 250

    gradient_accumulate_every = 2

    image_size = diffusion_model.image_size




if __name__ == '__main__':
    model = UNet224(3,3,64,256).cuda()
    input = torch.randn((16, 3, 224, 224)).cuda()
    T = 1000
    t = torch.randint(0, T, (16,)).cuda()
    output = model(input,t)
