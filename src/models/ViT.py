import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class PatchEmbedding(nn.Module):
    """将图像分割为Patch并嵌入为向量"""

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # 使用卷积层实现Patch分割和嵌入
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: [B, C, H, W] -> [B, embed_dim, num_patches^0.5, num_patches^0.5]
        x = self.proj(x)
        # 展平为序列 [B, embed_dim, num_patches] -> [B, num_patches, embed_dim]
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制"""

    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"

        # Q, K, V 投影矩阵
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape  # N: 序列长度（num_patches + 1）

        # 生成Q, K, V [B, N, 3*embed_dim] -> [3, B, num_heads, N, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 各 [B, num_heads, N, head_dim]

        # 计算注意力分数 [B, num_heads, N, N]
        attn = (q @ k.transpose(-2, -1)) * (1.0 / sqrt(self.head_dim))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # 加权求和 [B, num_heads, N, head_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x


class MLP(nn.Module):
    """Transformer中的前馈网络"""

    def __init__(self, embed_dim=768, hidden_dim=3072, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """Transformer编码器块"""

    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)

    def forward(self, x):
        # 残差连接 + 层归一化 + 自注意力
        x = x + self.attn(self.norm1(x))
        # 残差连接 + 层归一化 + 前馈网络
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """完整的ViT模型"""

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_channels=3,
            num_classes=1000,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            dropout=0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        # 可学习的类别标记和位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)

        # Transformer编码器堆叠
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # 分类头
        self.fc = nn.Linear(embed_dim, num_classes)

        # 初始化权重
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]  # Batch size

        # Patch嵌入 [B, num_patches, embed_dim]
        x = self.patch_embed(x)

        # 添加类别标记 [B, 1, embed_dim] -> [B, num_patches+1, embed_dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # 添加位置编码并Dropout
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        # 通过Transformer编码器
        for block in self.blocks:
            x = block(x)

        # 取类别标记对应的输出作为分类特征
        x = self.norm(x)
        cls_output = x[:, 0]

        # 分类头
        logits = self.fc(cls_output)
        return logits


# 预定义ViT模型配置（ViT-Base为例）
def vit_base_patch16_224(par,num_classes=1000):
    return VisionTransformer(
        img_size=par.img_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        num_classes=num_classes,
    )


