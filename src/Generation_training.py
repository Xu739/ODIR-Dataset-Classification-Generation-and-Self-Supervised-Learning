import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import random

import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from torch import nn, optim
import yaml
from torchvision import utils
from tqdm import tqdm
import torch.nn.functional as F
global device

os.environ["WANDB_MODE"] = "offline"
sys.path.append(str(Path(__file__).parent.parent))

#Change script path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(root_dir)
print(root_dir)
from src.models.model import GetModel
from src.models.vqvae_transformer import VQVAE, TokenTransformer, save_image_grid, train_vqvae, train_transformer, sample_images
from src.models.ddim import q_sample, compute_fid, sample_ddim_cfg, linear_beta_schedule
from experiments.configs.par import Struct
from src.utils import GetCriterion,GetOptim

from data.dataset_utils import GetDataset, GetTransform



def train_vqvae_transformer(model, train_dataloader, val_dataloader, par, save_dir, device):
    # ----------------------------
    # train vqvae
    # ----------------------------

    vqvae = VQVAE(codebook_size=par.codebook_size,D = par.D).to(device)

    optim_vqvae, _ = GetOptim(par,vqvae)

    train_vqvae(vqvae, train_dataloader, optim_vqvae, save_dir, par, device)

    # ----------------------------
    # train transformer
    # ---------------------------

    vq_state = torch.load(os.path.join(save_dir, 'vqvae_best.pt'))
    vqvae.load_state_dict(vq_state)
    vqvae.eval()

    gpt = TokenTransformer(vocab_size=par.codebook_size, seq_len=par.seq_len,
                           d_model=par.gpt_d_model, n_head=par.gpt_heads,
                           n_layer=par.gpt_layers, dropout=par.gpt_dropout).to(device)
    optim_gpt = torch.optim.AdamW(gpt.parameters(), lr=par.lr, betas=(0.9, 0.95), weight_decay=0.01)

    train_transformer(vqvae, gpt, train_dataloader, optim_gpt, save_dir, par, device)


    # ----------------------------
    # sampling
    # ---------------------------

    sample_images(None,None,save_dir,True)



def train_ddim_cfg(model, train_dataloader, val_dataloader, par, save_dir, device):
    betas = linear_beta_schedule(par.T)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0).to(device)
    optimizer, scheduler = GetOptim(par, model)
    mse = nn.MSELoss()

    start_epoch = 0
    if par.is_resume:
        path = f'/home/xukaijie/zju/checkpoints_ddim/model_epoch_{par.start_epoch}.pt'
        state = torch.load(path)
        model.load_state_dict(state)
        start_epoch = par.start_epoch

    for epoch in range(start_epoch,par.epochs):
        for step, (x, y) in enumerate(train_dataloader):
            x = x.to(device)
            y = y.to(device)

            if torch.rand(1).item() < par.drop_prob:
                y = None

            b = x.shape[0]
            t = torch.randint(0, par.T, (b,), device=device).long()
            noise = torch.randn_like(x).to(device)
            x_t = q_sample(x, t, noise,alphas_cumprod)

            t_norm = t.float() / par.T
            pred = model(x_t.float(), t_norm.float().to(device), y)

            loss = mse(pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")
        scheduler.step()
                # 保存 checkpoint
        ckpt_path = os.path.join(save_dir, f"model_epoch_{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)
        # 训练结束后
        fid_score = compute_fid(model, train_dataloader, device,par, n_samples=2000)
        print("FID:", fid_score)
        # 每个 epoch 自动采样
        for label in range(8):
            samples = sample_ddim_cfg(model, label,par,device,alphas, alphas_cumprod,n_samples=8, sample_steps=50)
            utils.save_image(samples, os.path.join(save_dir, f"samples_epoch_{epoch}_{label}.png"), nrow=4)
        print(f"[INFO] Epoch {epoch} checkpoint & samples saved.")


def train_cGAN(model, train_dataloader, val_dataloader, par, save_dir, device):
    def seed_torch(seed=2021):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    seed_torch()
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
    netD.apply(weights_init)
    netG.apply(weights_init)
    lb = LabelBinarizer()
    lb.fit(list(range(0, par.num_classes)))

    # 将标签进行one-hot编码
    def to_categrical(y: torch.FloatTensor):
        y_one_hot = lb.transform(y.cpu())
        floatTensor = torch.FloatTensor(y_one_hot)
        return floatTensor.to(device)



    print('netG:', '\n', netG)
    print('netD:', '\n', netD)

    print('training on:   ', device, '   start_epoch', start_epoch)

    netD, netG = netD.to(device), netG.to(device)

    criterion = GetCriterion(par,None)
    criterion.to(device)

    optimizerD = optim.Adam(netD.parameters(), lr=par.d_lr,betas= (0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=par.g_lr,betas= (0.5, 0.999))



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

            data = data.to(device)
            target = target.to(device)

            # 拼接真实数据和标签
            target1 = to_categrical(target).float()  # 加到噪声上
            target1_ = target1.unsqueeze(2).unsqueeze(3)
            target2 = target1_.repeat(1, 1, data.size(2), data.size(3))  # 加到数据上
            data = torch.cat((data, target2),
                             dim=1)  #  (N,nc,256,256),(N,n_classes, 256,256)->(N,nc+nc_classes,256,2568)

            label = torch.full((data.size(0), 1), real_label).to(device)

            # （1）训练判别器
            # training real data
            netD.zero_grad()
            output = netD(data)




            loss_D1 = criterion(output, label)
            loss_D1.backward()

            # training fake data,拼接噪声和标签
            noise_z = torch.randn(data.size(0), par.latent_dim, 1).to(device)
            noise_z = torch.cat((noise_z, target1.unsqueeze(2)), dim=1)  # (N,nz+n_classes,1,1)
            # 拼接假数据和标签
            fake_data = netG(noise_z)
            # print(fake_data.shape)
            fake_data = torch.cat((fake_data, target2), dim=1)
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

        # torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
        torch.save({'netG': netG.state_dict(),'netD': netD.state_dict(),},os.path.join(save_dir, f'model_epoch_{epoch}.pth'))
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
    subprocess.call(f'cp ./experiments/configs/Generation.yaml {save_dir}', shell=True)  # Save config

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
        train_cGAN(model, train_dataloader, val_dataloader, par, save_dir, device)
    elif par.model in ['ddim']:
        train_ddim_cfg(model.to(device), train_dataloader, val_dataloader, par, save_dir, device)
    elif par.model in ['vqvae']:
        train_vqvae_transformer(model, train_dataloader, val_dataloader, par, save_dir, device)
