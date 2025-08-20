import argparse
import os

import sys
from pathlib import Path
import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sympy.core.random import choice
from torch import nn, optim
import yaml
from torch.nn import DataParallel
from torchvision import utils
from tqdm import tqdm
import torch.nn.functional as F



global device
from torch_fidelity import calculate_metrics
from torchvision.utils import save_image
sys.path.append(str(Path(__file__).parent.parent))
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix,
    balanced_accuracy_score
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
#Change script path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(root_dir)
print(root_dir)
from src.models.model import GetModel
from src.models.vqvae_transformer import VQVAE, TokenTransformer, save_image_grid, train_vqvae, train_transformer, sample_images
from src.models.ddim import q_sample, compute_fid, sample_ddim_cfg, linear_beta_schedule
from src.models.resnet import resnet50
from experiments.configs.par import Struct
from src.Classification_training import plot_ovo_confusion_matrix
from src.utils import GetCriterion,GetOptim

from data.dataset_utils import GetDataset, GetTransform
def evaluate_fid_kid_is(real_loader, gen_root, device):
    fid = FrechetInceptionDistance(normalize=True).to(device)
    kid = KernelInceptionDistance(subset_size = 1000,normalize=True).to(device)
    is_metric = InceptionScore(normalize=True).to(device)


    # real_dataset = datasets.ImageFolder(real_root, transform=transforms.ToTensor())
    gen_dataset = datasets.ImageFolder(gen_root, transform=transforms.ToTensor())


    # real_loader = DataLoader(real_dataset, batch_size=32, shuffle=False)
    gen_loader = DataLoader(gen_dataset, batch_size=32, shuffle=False)


    for x, _ in tqdm(real_loader, desc="FID/KID: real"):
        fid.update(x.to(device), real=True)
        kid.update(x.to(device), real=True)


    for x, _ in tqdm(gen_loader, desc="FID/KID/IS: gen"):
        fid.update(x.to(device), real=False)
        kid.update(x.to(device), real=False)
        is_metric.update(x.to(device))


    fid_val = fid.compute().item()
    kid_mean, kid_std = kid.compute()
    is_mean, is_std = is_metric.compute()


    print(f"FID: {fid_val:.4f}")
    print(f"KID: {kid_mean:.4f} ± {kid_std:.4f}")
    print(f"IS: {is_mean:.4f} ± {is_std:.4f}")

@torch.no_grad()
def generate_cgan_by_class_onehot(netG, n_classes, samples_per_class, gen_root, device, latent_dim):
    """
    按类别生成固定数量样本（one-hot标签）并保存
    Args:
        netG: 训练好的生成器
        n_classes: 类别数量
        samples_per_class: 每个类别生成的样本数量
        gen_root: 图片保存根目录
        device: 运行设备
        latent_dim: 潜在向量维度
    """
    netG.eval()
    os.makedirs(gen_root, exist_ok=True)
    batch_size = 128
    with torch.no_grad():
        for c in range(n_classes):
            class_dir = os.path.join(gen_root, f"class_{c}")
            os.makedirs(class_dir, exist_ok=True)

            # 计算需要多少个批次
            num_batches = (samples_per_class + batch_size - 1) // batch_size
            generated_count = 0

            print(f"[INFO] Generating class {c}: {samples_per_class} samples in {num_batches} batches")

            for batch_idx in range(num_batches):
                # 计算当前批次的样本数量
                current_batch_size = min(batch_size, samples_per_class - generated_count)

                # 标签 one-hot
                labels = F.one_hot(
                    torch.full((current_batch_size,), c, dtype=torch.long, device=device),
                    num_classes=n_classes
                ).float()  # (N, n_classes)
                labels_exp = labels.unsqueeze(2)  # (N, n_classes, 1)

                # 噪声
                noise_z = torch.randn(current_batch_size, latent_dim, 1, device=device)

                # 拼接噪声和 one-hot 标签
                input_G = torch.cat((noise_z, labels_exp), dim=1)  # (N, latent_dim + n_classes, 1)

                # 生成图像
                gen_imgs = netG(input_G)
                gen_imgs = (gen_imgs.clamp(-1, 1) + 1) / 2  # [-1,1] -> [0,1]

                # 保存当前批次的图片
                for i in range(current_batch_size):
                    img_idx = generated_count + i
                    save_path = os.path.join(class_dir, f"{img_idx}.png")
                    save_image(gen_imgs[i], save_path)

                generated_count += current_batch_size

                # 打印进度
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                    print(
                        f"  Batch {batch_idx + 1}/{num_batches}: Generated {generated_count}/{samples_per_class} samples")
            print(f"[INFO] Class {c}: Completed {generated_count} samples")

from tqdm import tqdm
# -------------------------------
# DDIM 生成函数 (带 CFG)
# -------------------------------
def generate_images_ddim_cfg(model, gen_root, Config, device,
                             alphas, alphas_cumprod,
                             n_samples_per_class=100,
                             batch_size=32,
                             num_classes=8,
                             sample_steps=50,
                             eta=0.0,
                             guidance_scale=3.0):
    """
    用 DDIM+CFG 生成图像，并保存到 gen_root/class_x/ 目录
    """






    model.eval()
    os.makedirs(gen_root, exist_ok=True)
    label = 0
    n_generated = 0
    # pbar = tqdm(range(num_classes),desc=f'{label} is generated {n_generated}/{n_samples_per_class}')
    for label in tqdm(range(num_classes)):
        save_dir = os.path.join(gen_root, f"class_{label}")
        os.makedirs(save_dir, exist_ok=True)

        n_generated = 0
        with torch.no_grad():
            while n_generated < n_samples_per_class:
                # pbar.set_postfix()
                cur_bs = min(batch_size, n_samples_per_class - n_generated)

                imgs = sample_ddim_cfg(
                    model, label, Config, device,
                    alphas, alphas_cumprod,
                    n_samples=cur_bs,
                    sample_steps=sample_steps,
                    eta=eta,
                    guidance_scale=guidance_scale,

                )
                # 保存图片
                for k in range(cur_bs):
                    save_path = os.path.join(save_dir, f"{n_generated+k:05d}.png")
                    save_image(imgs[k], save_path)
                n_generated += cur_bs
                # postfix_str = f'{label}: {n_generated}/{n_samples_per_class}'
                # pbar.set_postfix_str(postfix_str)
                # pbar.update(cur_bs)
        # pbar.update(1)
        print(f"✅ 生成完成: class {label}, 共 {n_samples_per_class} 张")

    print(f"🎉 所有类别图像已生成到 {gen_root}")

def evaluate_classifier(classifier, gen_root, device, num_classes=8, image_size=128,is_wandb=False):
    transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    gen_dataset = datasets.ImageFolder(gen_root, transform=transform)
    gen_loader = DataLoader(gen_dataset, batch_size=64, shuffle=False)


    classifier.eval()
    test_loss = 0
    all_labels = []
    all_preds = []


    with torch.no_grad():
        for x, y in tqdm(gen_loader, desc="Classifier eval"):
            img = x.cuda()
            label = y.cuda()

            output = classifier(img)


            preds = torch.argmax(output, dim=1)
            all_labels.append(label.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

        # Combine results
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    # Default class names for ODIR dataset

    class_names = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']

    # Calculate metrics
    metrics = {

        'accuracy': accuracy_score(all_labels, all_preds),
        'balanced_accuracy': balanced_accuracy_score(all_labels, all_preds),
        'confusion_matrix': confusion_matrix(all_labels, all_preds),
        'classification_report': classification_report(
            all_labels, all_preds, target_names=class_names, output_dict=True
        )
    }
    print(classification_report(
            all_labels, all_preds, target_names=class_names, output_dict=False
        ))
    plot_ovo_confusion_matrix(metrics['confusion_matrix'], class_names, normalize=True, save_dir=gen_root,is_wandb=False)


def evluate_cGAN(model,classifier, gen_root,par,real_loader,device):
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    evaluate_fid_kid_is(real_loader, gen_root, device)
    evaluate_classifier(classifier, gen_root, device, num_classes=8, image_size=224)

def evluate_ddim(model,classifier, gen_root,par,real_loader,device):
    # transform = GetTransform(par,'test')
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    evaluate_fid_kid_is(real_loader, gen_root, device)
    evaluate_classifier(classifier, gen_root, device,num_classes=8,image_size=224)


# 修改你的加载代码，在 load_state_dict 之前处理权重
def load_state_dict_without_module( state_dict):
    """移除state_dict中的module前缀"""
    from collections import OrderedDict
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        # 移除 'module.' 前缀
        if k.startswith('module.'):
            name = k[7:]  # 去掉 'module.'
        else:
            name = k
        new_state_dict[name] = v

    return new_state_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parameter parser for model training and generation')

    # Required parameters
    parser.add_argument('--model', type=str, required=True,
                        choices=['cGAN', 'ddim', 'vqvae'],
                        help='Model name, options: cGAN, ddim, vqvae')

    # Optional parameters
    parser.add_argument('--num_classes', type=int, default=8,
                        )
    parser.add_argument('--par_path', type=str, default='./experiments/log/20250820-105330/Generation.yaml',
                        help='Path to parameter configuration file')
    parser.add_argument('--gen_root', type=str, default='./data/fake_figure/',
                        help='Path to save generated data')
    parser.add_argument('--real_root', type=str, default='./data/ODIR/ODIR-5K/Training Images',
                        help='Path to real data directory')
    parser.add_argument('--classifier_path', type=str, default='./experiments/log/20250819-224014/checkpoint_77.pth',
                        help='Path to classifier model')
    parser.add_argument('--model_path', type=str,
                        default='./ODIR_project/experiments/log/20250820-105330/model_epoch_184.pth',
                        help='Path to main model')
    parser.add_argument('--device', type=str, default=None,
                        help='Device specification, e.g.: cuda:0, cuda:1, cpu. Auto-select if not specified')
    parser.add_argument('--latent_dim', type=int, default=100,
                       )
    parser.add_argument('--debug_shapes', type=bool, default=False,
                        )
    parser.add_argument('--img_size', type=int, default=128,
                        )
    parser.add_argument('--T', type=int, default=1000,
                        )
    args = parser.parse_args()

    model_name = args.model # ['cGAN', 'ddim','vqvae']



    par_path = args.par_path
    gen_root = args.gen_root
    # real_root = './data/ODIR/ODIR-5K/Training Images'
    classifier_path = args.classifier_path
    model_path =args.model_path
    device = args.device


    classifier = resnet50(num_classes=8)
    if model_name == 'ddim':
        img_size = 128
    else:
        img_size = 256
    gen_root = os.path.join(gen_root, model_name)

    classifier_state = torch.load(classifier_path)
    par = yaml.safe_load(open(par_path, 'r'))
    par = Struct(**par)
    dataset, dataloader = GetDataset(par,'train',transform=transforms.Compose([
        transforms.Resize((img_size,img_size),),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3,std=[0.5]*3)


    ]),is_val=True)
    classifier.load_state_dict(load_state_dict_without_module(classifier_state))
    classifier = classifier.cuda()
    if model_name == 'cGAN':
        model = GetModel(args)
        need_generate = (not os.path.exists(gen_root)) or (len(os.listdir(gen_root)) == 0)
        netG = model.netG.model.to(device)
        state = torch.load(model_path)['netG']
        netG.load_state_dict(state)
        if need_generate:
            print(f"⚠️  {gen_root} is empty")
            generate_cgan_by_class_onehot(netG, args.num_classes, 5000, gen_root, device, args.latent_dim)
        else:
            print(f"✅  {gen_root} conclude figures")
        evluate_cGAN(netG,classifier,gen_root,par,dataloader,device)



    elif model_name == 'ddim':
        betas = linear_beta_schedule(par.T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0).to(device)
        model = GetModel(args)
        model.load_state_dict(load_state_dict_without_module(torch.load(model_path)))

        need_generate = (not os.path.exists(gen_root)) or (len(os.listdir(gen_root)) == 0)
        if need_generate:
            print(f"⚠️ 检测到 {gen_root} 为空，自动调用 DDIM 生成图像...")
            generate_images_ddim_cfg(
                model.to(device), gen_root,alphas=alphas,alphas_cumprod=alphas_cumprod,Config=args,
                n_samples_per_class=5000,
                device=device
            )
        else:
            print(f"✅ 检测到 {gen_root} 已有生成图像，跳过生成步骤")
        evluate_ddim(model,classifier,gen_root,par,dataloader,device)

    elif model_name == 'vqvae':
        pass
