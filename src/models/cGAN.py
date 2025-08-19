import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import par


# 梯度惩罚项
def compute_gradient_penalty(critic, real_samples, fake_samples, labels):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).cuda()
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = critic(interpolates,labels)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# 判别器损失
# def train_critic(real_imgs):
    # z = torch.randn(real_imgs.shape[0], latent_dim).cuda()
    # fake_imgs = generator(z)
    # real_scores = critic(real_imgs)
    # fake_scores = critic(fake_imgs.detach())
    # gp = compute_gradient_penalty(critic, real_imgs, fake_imgs)
    # loss = -torch.mean(real_scores) + torch.mean(fake_scores) + lambda_gp * gp
    # return loss

# 生成器损失
# def train_generator():
#     z = torch.randn(batch_size, latent_dim).to(device)
#     fake_imgs = generator(z)
#     loss = -torch.mean(critic(fake_imgs))
#     return loss
class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10, img_size=224, par=None):
        super().__init__()
        self.img_size = img_size
        self.label_embedding = nn.Embedding(num_classes, latent_dim)

        # 计算初始特征图大小 (经过多次上采样后达到224x224)
        self.init_size = img_size // 32  # 初始尺寸：7x7 (当img_size=224时)

        self.l1 = nn.Sequential(
            nn.Linear(latent_dim * 2, 512 * self.init_size ** 2),
            nn.BatchNorm1d(512 * self.init_size ** 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        out_channels = 1 if par.dataset == 'mnist' else 3
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 14x14
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 28x28
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 56x56
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 112x112
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(32, out_channels, 4, 2, 1),  # 224x224
            nn.Tanh()
        )
        # noise = torch.randn(latent_dim)

    def forward(self, z, labels):
        # 融合噪声和标签
        c = self.label_embedding(labels)
        # z = z * c  # 元素级相乘
        z = torch.cat((z, c), dim=1)


        out = self.l1(z)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        img = self.conv_blocks(out)

        #确保输出尺寸正确
        # if img.size(-1) != self.img_size:
        #     img = F.interpolate(img, size=(self.img_size, self.img_size))
        # print(img.shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, num_classes=10, img_size=224,par=None):
        super().__init__()

        self.in_channels = 1 if par.dataset == 'mnist' else 3

        self.num_classes = num_classes


        self.label_embedding = nn.Embedding(num_classes, num_classes * img_size * img_size)
        self.img_size = img_size
        ndf = 64
        # self.model = nn.Sequential(
        #     # 输入通道：6 (3图像 + 3标签嵌入)
        #     nn.Conv2d(2*self.out_channels, 64, 4, 2, 1),  # 112x112
        #     nn.LeakyReLU(0.2, inplace=True),
        #
        #     nn.Conv2d(64, 128, 4, 2, 1),  # 56x56
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(0.2, inplace=True),
        #
        #     nn.Conv2d(128, 256, 4, 2, 1),  # 28x28
        #     nn.BatchNorm2d(256),
        #     nn.LeakyReLU(0.2, inplace=True),
        #
        #     nn.Conv2d(256, 512, 4, 2, 1),  # 14x14
        #     nn.BatchNorm2d(512),
        #     nn.LeakyReLU(0.2, inplace=True),
        #
        #     nn.Conv2d(512, 512, 4, 2, 1),  # 7x7
        #     nn.BatchNorm2d(512),
        #     nn.LeakyReLU(0.2, inplace=True),
        #
        #     # nn.AdaptiveAvgPool2d((1, 1)),
        #
        #     nn.Flatten(),
        #     nn.Linear(512 *( img_size // 32) * (img_size//32), 512),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, 1),
        #     nn.Sigmoid()
        #
        # )
        self.model = nn.Sequential(nn.Conv2d(self.in_channels + num_classes, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2),
                      nn.LeakyReLU(0.2, inplace=True),
                      nn.Conv2d(ndf * 2, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2),
                      nn.LeakyReLU(0.2, inplace=True),
                      nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4),
                      nn.LeakyReLU(0.2, inplace=True),
                      nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4),
                      nn.LeakyReLU(0.2, inplace=True),
                      nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 8),
                      nn.LeakyReLU(0.2, inplace=True),
                      nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),  # (N,1,1,1)
                      nn.Flatten(),  # (N,1)
                      nn.Sigmoid()
                      )
    def forward(self, img, labels):
        # 将标签嵌入调整为图像形状并拼接

        c = self.label_embedding(labels).view(img.shape[0], self.num_classes, self.img_size, self.img_size)
        x = torch.cat([img, c], dim=1)
        return self.model(x)

class cGAN(nn.Module):
    def __init__(self,par):
        super().__init__()
        self.img_size = par.img_size
        self.latent_dim = par.latent_dim
        self.generator = Generator(self.latent_dim, par.num_classes,self.img_size)
        self.discriminator = Discriminator(par.num_classes, self.img_size)
