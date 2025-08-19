import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import wandb
from torch import nn


os.environ["WANDB_MODE"] = "offline"
sys.path.append(str(Path(__file__).parent.parent))

#Change script path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(root_dir)
print(root_dir)
from src.models.model import GetModel
import yaml
from experiments.configs.par import Struct


from data.dataset_utils import GetDataset, GetTransform

if __name__ == '__main__':
    # Load configuration
    path_yaml = "./experiments/configs/Generation.yaml"
    par = yaml.safe_load(open(path_yaml, 'r'))

    # Create save directory for checkpoints
    save_dir = f'./experiments/log/{datetime.now().strftime("%Y%m%d-%H%M%S")}/'
    os.makedirs(save_dir, exist_ok=True)
    subprocess.call(f'cp ./experiments/configs/Classification.yaml {save_dir}', shell=True)  # Save config

    # Initialize wandb
    wandb.init(project="ODIR", name=f"{save_dir}", config=par)
    par = Struct(**par)
    os.environ["CUDA_VISIBLE_DEVICES"] = par.CUDA_VISIBLE_DEVICES

    # Data transformations
    train_transform = GetTransform(par, 'train')
    test_transform = GetTransform(par, 'test')

    # Get datasets and dataloaders
    train_dataset, train_dataloader = GetDataset(par, 'train', train_transform)
    val_dataset, val_dataloader = GetDataset(par, 'val', test_transform)
    test_dataset, test_dataloader = GetDataset(par, 'test', test_transform)

    # Initialize model
    model = GetModel(par)
    model = nn.DataParallel(model).cuda()

    # print(model)

