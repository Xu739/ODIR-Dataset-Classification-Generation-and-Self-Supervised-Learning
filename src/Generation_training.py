import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from torch import nn, optim
import yaml
global device

os.environ["WANDB_MODE"] = "offline"
sys.path.append(str(Path(__file__).parent.parent))

#Change script path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(root_dir)
print(root_dir)
from src.models.model import GetModel

from experiments.configs.par import Struct
from src.utils import GetCriterion,GetOptim

from data.dataset_utils import GetDataset, GetTransform


def train_cGAN(model, train_dataloader, val_dataloader, par, save_dir):
    # real label
    real_label = 1.0
    # fake label
    fake_label = 0.0
    start_epoch = 0

    netD = model.netD.model
    netG = model.netG.model

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.zeros_(m.bias)

    lb = LabelBinarizer()
    lb.fit(list(range(0, par.num_classes)))

    # 将标签进行one-hot编码
    def to_categrical(y: torch.FloatTensor):
        y_one_hot = lb.transform(y.cpu())
        floatTensor = torch.FloatTensor(y_one_hot)
        return floatTensor.to(device)

    # 样本和one-hot标签进行连接，以此作为条件生成
    def concanate_data_label(data, y):  # data （N,nc, 128,128）
        y_one_hot = to_categrical(y)  # (N,1)->(N,n_classes)

        con = torch.cat((data, y_one_hot), 1)

        return con

    print('netG:', '\n', netG)
    print('netD:', '\n', netD)

    print('training on:   ', device, '   start_epoch', start_epoch)

    netD, netG = netD.to(device), netG.to(device)

    criterion = GetCriterion(par,None)
    criterion.to(device)

    optimizerD = optim.Adam(netD.parameters(), lr=par.d_lr)
    optimizerG = optim.Adam(netG.parameters(), lr=par.g_lr)



    def gen_img_plot(model, text_input, labels, n_classes, save_dir,device, epoch):

        labels_eb = nn.Embedding(n_classes, n_classes).to(device)(labels)
        input = torch.cat((text_input, labels_eb
                           .reshape(n_classes, -1, 1)), 1)
        n_samples = n_classes
        prediction = np.squeeze(model(input).detach().cpu().numpy()[:n_samples])
        os.makedirs(save_dir, exist_ok=True)
        # fig, axs = plt.subplots(1, n_samples, figsize=(20, 2))
        fig, axs = plt.subplots(1, n_samples, figsize=(n_classes, 2))
        # labels.reshape(10,1)
        for i in range(n_samples):
            axs[i].imshow(np.transpose((prediction[i] + 1) / 2, (1, 2, 0)))  # CHW -> HWC
            axs[i].set_title(f"{labels[i].item()}")
            axs[i].axis('off')
        plt.savefig(os.path.join(save_dir, f"samples_epoch_{epoch}.png"))
        plt.show()

        plt.close()

    for epoch in range(start_epoch, 500):
        for batch, (data, target) in enumerate(train_dataloader):
            #         if epoch%2==0 and batch==0:
            #             torchvision.utils.save_image(data[:16], filename='./generated_fake/%s/源epoch_%d_grid.png'%(datasets,epoch),nrow=4,normalize=True)
            data = data.to(device)
            target = target.to(device)

            # 拼接真实数据和标签
            target1 = to_categrical(target).float()  # 加到噪声上
            target1_ = target1.unsqueeze(2).unsqueeze(3)
            target2 = target1_.repeat(1, 1, data.size(2), data.size(3))  # 加到数据上
            data = torch.cat((data, target2),
                             dim=1)  # 将标签与数据拼接 (N,nc,128,128),(N,n_classes, 128,128)->(N,nc+nc_classes,128,128)

            label = torch.full((data.size(0), 1), real_label).to(device)

            # （1）训练判别器
            # training real data
            netD.zero_grad()
            output = netD(data)

            D_loss = 0


            loss_D1 = criterion(output, label)
            loss_D1.backward()

            # training fake data,拼接噪声和标签
            noise_z = torch.randn(data.size(0), par.latent_dim, 1).to(device)
            noise_z = torch.cat((noise_z, target1.unsqueeze(2)), dim=1)  # (N,nz+n_classes,1,1)
            # 拼接假数据和标签
            fake_data = netG(noise_z)
            # print(fake_data.shape)
            fake_data = torch.cat((fake_data, target2), dim=1)  # (N,nc+n_classes,128,128)
            label = torch.full((data.size(0), 1), fake_label).to(device)

            output = netD(fake_data.detach())


            loss_D2 = criterion(output, label)
            loss_D2.backward()

            # 更新判别器
            optimizerD.step()







            netG.zero_grad()
            label = torch.full((data.size(0), 1), real_label).to(device)

            output = netD(fake_data.to(device))

            lossG = criterion(output, label)
            lossG.backward()

            # 更新生成器
            optimizerG.step()

            if batch % 10 == 0:

                print('epoch: %4d, batch: %4d, discriminator loss: %.4f, generator loss: %.4f'
                      % (epoch, batch, loss_D1.item() + loss_D2.item(), lossG.item()))
        noise_z1 = torch.randn(par.num_classes, par.latent_dim, 1).to(device)
        text_labels = torch.tensor([i for i in range(par.num_classes)]).reshape(par.num_classes, ).to(device)
        gen_img_plot(netG, noise_z1, text_labels, par.num_classes, save_dir, device,epoch)

if __name__ == '__main__':
    # Load configuration
    path_yaml = "./experiments/configs/Generation.yaml"
    par = yaml.safe_load(open(path_yaml, 'r'))

    # Create save directory for checkpoints
    save_dir = f'./experiments/log/{datetime.now().strftime("%Y%m%d-%H%M%S")}/'
    os.makedirs(save_dir, exist_ok=True)
    subprocess.call(f'cp ./experiments/configs/Classification.yaml {save_dir}', shell=True)  # Save config

    global device

    # Initialize wandb
    wandb.init(project="ODIR", name=f"{save_dir}", config=par)
    par = Struct(**par)
    device = torch.device(f"cuda:{par.CUDA_VISIBLE_DEVICES}" if torch.cuda.is_available() else "cpu")
    # os.environ["CUDA_VISIBLE_DEVICES"] = par.CUDA_VISIBLE_DEVICES

    # Data transformations
    train_transform = GetTransform(par, 'train',is_generation=True)
    test_transform = GetTransform(par, 'test',is_generation=True)

    # Get datasets and dataloaders
    train_dataset, train_dataloader = GetDataset(par, 'train', train_transform)
    val_dataset, val_dataloader = GetDataset(par, 'val', test_transform)
    test_dataset, test_dataloader = GetDataset(par, 'test', test_transform)

    # Initialize model
    model = GetModel(par)
    # model = nn.DataParallel(model).cuda()

    # print(model)

    if par.model in ['cGAN']:
        train_cGAN(model, train_dataloader, val_dataloader, par, save_dir)

