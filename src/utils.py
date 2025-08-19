import torch
from torch import nn, optim
import os
from tqdm import tqdm
import numpy as np
import wandb
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, classification_report,
    confusion_matrix, multilabel_confusion_matrix,
    balanced_accuracy_score
)

from src.models.model import GetModel

def GetOptim(par, model):
    """
    Initialize optimizer and learning rate scheduler based on configuration.

    Args:
        par (object): Configuration parameters containing:
            - optim: Optimizer type ('adam', 'SGD', 'rsm')
            - lr: Learning rate
            - weight_decay: Weight decay factor
            - scheduler: Learning rate scheduler type
            - milestones: List of epoch milestones for MultiStepLR
            - gamma: Multiplicative factor for learning rate decay
        model (nn.Module): Model whose parameters to optimize

    Returns:
        tuple: (optimizer, scheduler) pair
    """
    # Initialize optimizer based on configuration
    if par.optim == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=par.lr,
            weight_decay=par.weight_decay
        )
    elif par.optim == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=par.lr,
            weight_decay=par.weight_decay,
            momentum=0.9,
            nesterov=True
        )
    elif par.optim == 'rsm':
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=par.lr,
            weight_decay=par.weight_decay
        )
    else:
        optimizer = None

    # Initialize learning rate scheduler
    if par.scheduler == 'MultistepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=par.milestones,
            gamma=par.gamma
        )
    else:
        scheduler = None

    return optimizer, scheduler


class BestRecorder:
    """
    Tracks and records the best model performance during training.
    Implements early stopping based on validation accuracy.
    """

    def __init__(self, par, is_ssp=False):
        """
        Initialize the best result recorder.

        Args:
            par (object): Configuration parameters containing:
                - earlystop: Number of epochs to wait before early stopping
            is_ssp (bool): Whether in self-supervised pretraining mode
        """
        self.is_ssp = is_ssp
        self.acc_text = 'rot/accuracy' if is_ssp else 'accuracy'
        self.best_result = {self.acc_text: 0.}

        if par.earlystop != 0:
            self.cnt = par.earlystop
            self.early_stop = par.earlystop

    def update(self, result):
        """
        Update the best result and check for early stopping condition.

        Args:
            result (dict): Dictionary containing current epoch metrics

        Returns:
            bool: Whether to trigger early stopping
        """
        if result[self.acc_text] > self.best_result[self.acc_text]:
            self.best_result = result
            self.cnt = self.early_stop
        else:
            self.cnt -= 1

        return False if self.cnt != 0 else True


def GetCriterion(par, class_weights):
    """
    Initialize the loss function based on configuration.

    Args:
        par (object): Configuration parameters containing:
            - criterion: Loss function type ('CrossEntropyLoss')
            - dataset: Dataset name (determines whether to use class weights)
        class_weights (tensor): Class weights for imbalanced datasets

    Returns:
        nn.Module: Initialized loss function
    """
    print('GetCriterion:', class_weights)

    if par.criterion == 'CrossEntropyLoss':
        if par.dataset != 'cifar-10' and class_weights is not None:
            return nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float))
        else:
            return nn.CrossEntropyLoss()
    elif par.criterion == 'BCELoss':
        return nn.BCELoss()
    else:
        return None


def rotation(imgs, par):
    """
    Generate rotated versions of input images with corresponding rotation labels.

    Args:
        imgs (tensor): Input images (batch_size, 3, img_size, img_size)
        par (object): Configuration parameters containing img_size

    Returns:
        tuple: (rotated_imgs, targets) where:
            - rotated_imgs: Stack of rotated images (batch_size*4, 3, img_size, img_size)
            - targets: Rotation labels (0=0°, 1=90°, 2=180°, 3=270°)
    """
    imgs = imgs.reshape((-1, 3, par.img_size, par.img_size))
    batch_size = imgs.shape[0]
    rotated_imgs = []
    targets = []

    for i in range(batch_size):
        img = imgs[i]

        # 0° rotation (original)
        rotated_imgs.append(img)
        targets.append(0)

        # 90° rotation
        rotated_90 = torch.rot90(img, k=1, dims=(-2, -1))
        rotated_imgs.append(rotated_90)
        targets.append(1)

        # 180° rotation
        rotated_180 = torch.rot90(img, k=2, dims=(-2, -1))
        rotated_imgs.append(rotated_180)
        targets.append(2)

        # 270° rotation
        rotated_270 = torch.rot90(img, k=3, dims=(-2, -1))
        rotated_imgs.append(rotated_270)
        targets.append(3)

    rotated_imgs = torch.stack(rotated_imgs, dim=0)
    targets = torch.LongTensor(targets).to(imgs.device)

    return rotated_imgs, targets


def RotSSP(train_dataloader, val_dataloader, save_dir, par):
    """
    Self-Supervised Pretraining using rotation prediction (RotNet approach).

    Args:
        train_dataloader (DataLoader): Training data loader
        val_dataloader (DataLoader): Validation data loader
        save_dir (str): Directory to save checkpoints
        par (object): Configuration parameters

    Returns:
        str: Path to the best model checkpoint
    """
    # Initialize training components
    best_recorder = BestRecorder(par, True)
    wandb.define_metric('rot/step')
    wandb.define_metric('rot/*', step_metric='rot/step')

    model = GetModel(par, num_classes=4).cuda()  # 4 classes for rotation angles
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=par.lr, weight_decay=par.weight_decay)

    # Create directory for rotation checkpoints
    rot_dir = os.path.join(save_dir, 'rot')
    os.makedirs(rot_dir, exist_ok=True)

    for epoch in range(par.epochs):
        model.train()
        total_train_loss = 0
        all_labels, all_preds = [], []

        # Training phase
        for img, _ in tqdm(train_dataloader):
            img = img.cuda()
            img, label = rotation(img, par)
            label = label.cuda()

            # Forward pass (handle GoogleNet auxiliary outputs)
            if par.model == 'GoogleNet':
                output, output1, output2 = model(img)
                output = output + 0.3 * output1 + 0.3 * output2
            else:
                output = model(img)

            # Compute loss and update
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            total_train_loss += loss.item()
            _, preds = torch.max(output, 1)
            all_labels.extend(label.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        # Calculate training accuracy
        train_accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

        # Validation phase
        model.eval()
        val_loss = 0
        val_labels, val_preds, val_probs = [], [], []

        with torch.no_grad():
            for img, _ in tqdm(val_dataloader, desc=f"Val Epoch {epoch}"):
                img = img.cuda()
                img, label = rotation(img, par)
                label = label.cuda()

                output = model(img)
                val_loss += criterion(output, label).item()

                # Collect validation predictions
                _, preds = torch.max(output, 1)
                probs = torch.softmax(output, dim=1)
                val_labels.extend(label.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())

        # Calculate validation metrics
        val_labels = np.array(val_labels)
        val_preds = np.array(val_preds)
        val_probs = np.array(val_probs)

        try:
            auc = roc_auc_score(val_labels, val_probs, multi_class='ovr', average='macro')
        except:
            auc = -1

        # Prepare results dictionary
        result = {
            'rot/loss': val_loss / len(val_dataloader),
            'rot/accuracy': np.mean(val_preds == val_labels),
            'rot/precision': precision_score(val_labels, val_preds, average='macro'),
            'rot/recall': recall_score(val_labels, val_preds, average='macro'),
            'rot/f1': f1_score(val_labels, val_preds, average='macro'),
            'rot/auc': auc,
            'rot/train_loss': total_train_loss / len(train_dataloader),
            'rot/train_accuracy': train_accuracy,
            'rot/step': epoch
        }

        # Log results and save checkpoint
        wandb.log(result)
        checkpoint_path = os.path.join(rot_dir, f'checkpoint_{epoch}.pth')
        torch.save(model.state_dict(), checkpoint_path)

        # Check for early stopping
        if best_recorder.update(result):
            return checkpoint_path

    return None