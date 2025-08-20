# Import standard libraries
import subprocess
from datetime import datetime
import os



# Import third-party libraries
import pandas as pd
import torch
import torchvision.transforms as T
from torch import nn
import yaml
import wandb
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix,
    balanced_accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
from pathlib import Path





# Set wandb to offline mode
# os.environ["WANDB_MODE"] = "offline"


def train_one_epoch(epoch, train_loader, model, criterion, optimizer):
    """
    Train the model for one epoch.

    Args:
        epoch (int): Current epoch number
        train_loader (DataLoader): Training data loader
        model (nn.Module): Model to train
        criterion (loss function): Loss function
        optimizer (optimizer): Optimizer

    Returns:
        tuple: (average loss, accuracy) for the epoch
    """
    total_loss = 0
    model.train()
    all_labels = []
    all_preds = []

    for img, label in tqdm(train_loader):
        img = img.cuda()
        label = label.cuda()

        # Handle GoogleNet's auxiliary outputs
        if par.model == 'GoogleNet':
            output, output1, output2 = model(img)
            output = output + 0.3 * output1 + 0.3 * output2
        else:
            output = model(img)

        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(output, 1)
        all_labels.extend(label.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    accuracy = np.mean(all_preds == all_labels)

    return total_loss / len(train_loader), accuracy


def val_one_epoch(epoch, val_dataloader, model, criterion):
    """
    Validate the model for one epoch.

    Args:
        epoch (int): Current epoch number
        val_dataloader (DataLoader): Validation data loader
        model (nn.Module): Model to validate
        criterion (loss function): Loss function

    Returns:
        dict: Dictionary containing validation metrics
    """
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    all_probs = []  # For AUC calculation

    with torch.no_grad():
        for img, label in tqdm(val_dataloader, desc=f"Val Epoch {epoch}"):
            img = img.cuda()
            label = label.cuda()

            output = model(img)
            loss = criterion(output, label)
            total_loss += loss.item()

            # Collect predictions and probabilities
            _, preds = torch.max(output, 1)
            probs = torch.softmax(output, dim=1)

            all_labels.extend(label.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Calculate metrics
    metrics = {
        'loss': total_loss / len(val_dataloader),
        'accuracy': np.mean(all_preds == all_labels),
        'precision': precision_score(all_labels, all_preds, average='macro'),
        'recall': recall_score(all_labels, all_preds, average='macro'),
        'f1': f1_score(all_labels, all_preds, average='macro'),
    }

    # Calculate AUC (handle cases where it can't be calculated)
    try:
        metrics['auc'] = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except:
        metrics['auc'] = -1

    return metrics


def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, criterion, save_dir, par):
    """
    Main training loop for the model.

    Args:
        model (nn.Module): Model to train
        train_dataloader (DataLoader): Training data loader
        val_dataloader (DataLoader): Validation data loader
        optimizer (optimizer): Optimizer
        scheduler (scheduler): Learning rate scheduler
        criterion (loss function): Loss function
        save_dir (str): Directory to save checkpoints
        par (Struct): Configuration parameters

    Returns:
        dict: Best validation results
    """
    # Set random seeds for reproducibility
    torch.manual_seed(par.seed)
    torch.cuda.manual_seed(par.seed)

    best_recorder = BestRecorder(par)
    model = model.cuda()

    for epoch in range(par.epochs):
        # Training phase
        train_loss, train_acc = train_one_epoch(epoch, train_dataloader, model, criterion, optimizer)
        scheduler.step()

        # Validation phase
        val_result = val_one_epoch(epoch, val_dataloader, model, criterion)

        # Combine and log results
        result = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'epoch': epoch,
            **val_result
        }

        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(save_dir, f'checkpoint_{epoch}.pth'))

        # Update best results and check for early stopping
        early_stop = best_recorder.update(result)
        wandb.log(result)

        print(result)
        if early_stop:
            break

    print('Best result:', best_recorder.best_result)
    return best_recorder.best_result


def plot_ovo_confusion_matrix(cm, class_names, figsize=(10, 8), normalize=True,save_dir = None,is_wandb=True):
    """
    Plot a confusion matrix in OVO (One-vs-One) style.

    Args:
        cm (np.array): Confusion matrix
        class_names (list): List of class names
        figsize (tuple): Figure size
        normalize (bool): Whether to normalize the matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False
    )

    plt.title('Confusion Matrix (Normalized)' if normalize else 'Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Log to wandb
    if is_wandb:
        wandb.log({"confusion_matrix_plot": wandb.Image(plt)})

    plt.tight_layout()
    plt.show()
    plt.savefig(f"{save_dir}/ovo_confusion_matrix.png")


def test(model, test_dataloader, best_result, save_dir, criterion=None, class_names=None):
    """
    Test the trained model on the test set.

    Args:
        model (nn.Module): Model to test
        test_dataloader (DataLoader): Test data loader
        best_result (dict): Best validation results (contains epoch info)
        save_dir (str): Directory where checkpoints are saved
        criterion (loss function): Optional loss function
        class_names (list): List of class names

    Returns:
        dict: Dictionary containing test metrics
    """
    # Load best model weights
    model.load_state_dict(
        torch.load(os.path.join(save_dir, f'checkpoint_{best_result["epoch"]}.pth')),

    )

    model.eval()
    test_loss = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for img, label in tqdm(test_dataloader, desc="Testing"):
            img = img.cuda()
            label = label.cuda()

            output = model(img)
            if criterion is not None:
                test_loss += criterion(output, label).item()

            preds = torch.argmax(output, dim=1)
            all_labels.append(label.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    # Combine results
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    # Default class names for ODIR dataset
    if class_names is None:
        class_names = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']

    # Calculate metrics
    metrics = {
        'loss': test_loss / len(test_dataloader) if criterion else None,
        'accuracy': accuracy_score(all_labels, all_preds),
        'balanced_accuracy': balanced_accuracy_score(all_labels, all_preds),
        'confusion_matrix': confusion_matrix(all_labels, all_preds),
        'classification_report': classification_report(
            all_labels, all_preds, target_names=class_names, output_dict=True
        )
    }

    # Log confusion matrix to wandb
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=all_labels,
            preds=all_preds,
            class_names=class_names,
            title="Confusion Matrix"
        )
    })

    # Print results
    print("\n===== Test Report =====")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    return metrics


if __name__ == '__main__':

    sys.path.append(str(Path(__file__).parent.parent))
    # Change script path
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(root_dir)
    print(root_dir)
    # Import local modules
    from data.dataset_utils import GetTransform, GetDataset
    from src.utils import GetOptim, BestRecorder, GetCriterion, RotSSP
    from experiments.configs.par import Struct
    from src.models.model import GetModel
    # Load configuration
    path_yaml = "./experiments/configs/Classification.yaml"
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
    train_transform = GetTransform(par,'train')
    test_transform = GetTransform(par,'test')


    # Get datasets and dataloaders
    train_dataset, train_dataloader = GetDataset(par, 'train', train_transform)
    val_dataset, val_dataloader = GetDataset(par, 'val', test_transform)
    test_dataset, test_dataloader = GetDataset(par, 'test', test_transform)

    # Initialize model
    model = GetModel(par)
    model = nn.DataParallel(model).cuda()

    # Load pretrained weights if specified
    if par.is_cifar_10 == 1:
        print('cifar-10 pretrained')
        new_state_dict = model.state_dict()
        cifar_10_state = torch.load(par.cifar_10_state_dir)
        for name, param in cifar_10_state.items():
            if not name.startswith('module.fc.'):  # Skip final layer
                new_state_dict[name] = param
        model.load_state_dict(new_state_dict, strict=False)

    # Self-supervised pretraining if enabled
    if par.is_RotSSP:
        print("Using RotSSP pretraining")
        best_dir = RotSSP(train_dataloader, val_dataloader, save_dir, par)
        ssp_state = torch.load(best_dir)
        new_state_dict = model.state_dict()
        for name, param in ssp_state.items():
            if not name.startswith('fc.'):  # Skip final layer
                new_state_dict[name] = param
        model.load_state_dict(new_state_dict, strict=False)

    # Initialize optimizer and loss function
    optimizer, scheduler = GetOptim(par, model)
    criterion = GetCriterion(par, train_dataset.class_weights if par.dataset != 'cifar-10' else None).cuda()

    # Train the model
    best_result = train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, criterion, save_dir, par)

    # Set class names based on dataset
    if par.dataset == 'ODIR':
        class_names = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
    else:  # CIFAR-10
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Test the model
    report = test(model, test_dataloader, best_result, save_dir, criterion, class_names)
    plot_ovo_confusion_matrix(report['confusion_matrix'], class_names, normalize=True, save_dir=save_dir)