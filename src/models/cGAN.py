import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from torch import nn, optim




class netG(nn.Module):

    """Generator Network for Conditional GAN

    Generates images from random noise and class conditioning.
    Uses transposed convolutions for upsampling.

    Args:
        nz (int): Size of the noise vector (latent dimension)
        n_classes (int): Number of classes for conditional generation
        ngf (int): Number of generator features (base channel count)
        nc (int): Number of output channels (3 for RGB images)
    """
    def __init__(self,nz=100, n_classes=8, ngf=64, nc=3):
        super(netG, self).__init__()
        self.model = nn.Sequential(
                                    nn.Linear(1,4),
                                    nn.LeakyReLU(0.2,inplace=True),
                                    nn.Unflatten(dim=2,unflattened_size=(2,2,)),
                     nn.ConvTranspose2d(nz + n_classes, ngf * 16, 3, 1, 0, bias=False), nn.BatchNorm2d(ngf * 16),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 8),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 4),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.ConvTranspose2d(ngf * 4, ngf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 4),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.ConvTranspose2d(ngf * 2, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
                     nn.Tanh()

                     )

        def forward(self, input):
            return self.model(input)



class netD(nn.Module):
    """Discriminator Network for Conditional GAN
    Generates images from random noise and class conditioning.
    Uses transposed convolutions for upsampling.
    Args:
        nz (int): Size of the noise vector (latent dimension)
        n_classes (int): Number of classes for conditional generation
        ndf (int): Number of discriminate features (base channel count)
        nc (int): Number of output channels (3 for RGB images)
        """

    def __init__(self, nz=100, n_classes=8, ndf=64, nc=3):
        # 判别器             #(N,nc, 256,256)
        super(netD, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(nc + n_classes, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Conv2d(ndf * 2, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 8),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Conv2d(ndf * 8, ndf * 16, 4, 1, 0, bias=False),  # (N,1,1,1)
nn.LeakyReLU(0.2, inplace=True),
                     nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
                     nn.Flatten(),  # (N,1)
                     nn.Linear(4,1),
                     nn.Sigmoid()
                     )

    def forward(self, input):
        return self.model(input)


class cGAN(nn.Module):
    def __init__(self, nz=100, n_classes=8, ngf=64, ndf=64,nc=3):
        super(cGAN, self).__init__()
        self.netG = netG(nz, n_classes, ndf, nc)
        self.netD = netD(nz, n_classes, ndf, nc)

