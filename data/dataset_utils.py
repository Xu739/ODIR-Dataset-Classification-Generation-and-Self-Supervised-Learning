import os
from collections import Counter
import numpy as np
import torchvision.datasets
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
from PIL import Image
from imblearn.over_sampling import RandomOverSampler


class ImgDataset(Dataset):
    """Custom PyTorch Dataset for handling image classification tasks.

    Supports:
    - Loading images from CSV-specified paths
    - Automatic class weight computation
    - Oversampling for imbalanced datasets
    - Custom transformations
    """

    def __init__(self, csv_path, img_path, split_fold, data_type, oversample, par, transform=None):
        """
        Initialize the dataset.

        Args:
            csv_path (str): Path to directory containing CSV files
            img_path (str): Base path to image files
            split_fold (int): Current fold number for cross-validation
            data_type (str): Type of data ('train', 'val', 'test')
            oversample (bool): Whether to perform oversampling
            par (object): Parameter object with configuration
            transform (callable, optional): Optional transform to be applied
        """
        super().__init__()
        # Default transform if none provided
        self.transform = transform if transform is not None else T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])

        # Load data from CSV
        df_path = os.path.join(csv_path, str(split_fold), f'{data_type}.csv')
        df = pd.read_csv(df_path)
        self.file_path = [os.path.join(img_path, i) for i in df['filename']]
        self.labels = np.array(df['label'])
        self.len = len(self.labels)

        # Compute class weights for imbalanced datasets
        self.class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(self.labels),
            y=self.labels
        )

        # Apply oversampling if requested (typically for training data)
        if oversample:
            self._oversample_data()

        print(f"Class distribution: {Counter(self.labels)}")

    def _oversample_data(self):
        """Perform random oversampling to balance class distribution."""
        label_counts = Counter(self.labels)
        max_count = max(label_counts.values())

        # Create oversampler with strategy to equalize all classes
        ros = RandomOverSampler(
            sampling_strategy={k: max_count for k in label_counts.keys()}
        )

        # Resample indices
        indices = np.arange(len(self.labels)).reshape(-1, 1)
        resampled_indices, _ = ros.fit_resample(indices, self.labels)

        # Update dataset with oversampled data
        self.file_path = [self.file_path[i] for i in resampled_indices.flatten()]
        self.labels = self.labels[resampled_indices.flatten()]
        self.len = len(self.labels)

        # Recompute class weights after oversampling
        self.class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(self.labels),
            y=self.labels
        )

    def __len__(self):
        """Return total number of samples in dataset."""
        return self.len

    def __getitem__(self, idx):
        """Get single sample by index.

        Returns:
            tuple: (image_tensor, label) pair
        """
        img = Image.open(self.file_path[idx]).convert('RGB')
        img = self.transform(img)
        return img, self.labels[idx]


def GetDataset(par, data_type, transform=None):
    """Factory function to create appropriate dataset and dataloader.

    Args:
        par (object): Configuration parameters object
        data_type (str): Type of data ('train', 'val', 'test')
        transform (callable, optional): Optional transform to be applied

    Returns:
        tuple: (dataset, dataloader) pair
    """
    # Determine if oversampling should be applied
    is_oversample = (data_type == 'train' and par.is_oversample == 1)

    # Handle standard torchvision datasets
    if par.dataset == 'cifar-10':
        dataset = CIFAR10(
            root='./data',
            train=(data_type == 'train'),
            download=True,
            transform=transform
        )
    elif par.dataset == 'mnist':
        dataset = torchvision.datasets.MNIST(
            root='./data',
            train=(data_type == 'train'),
            download=True,
            transform=transform
        )
    # Handle custom dataset

    else:
        csv_path = os.path.join(
            par.datafile_path,
            par.dataset,
            'csv',
            f'{par.split_num - 2}11'
        )
        dataset = ImgDataset(
            csv_path,
            par.img_path,
            par.split_id,
            data_type,
            is_oversample,
            par,
            transform
        )

    # Create dataloader with configured parameters
    dataloader = DataLoader(
        dataset,
        batch_size=par.batch_size,
        shuffle=True,
        num_workers=par.num_workers,
        pin_memory=True,
        prefetch_factor=2
    )

    return dataset, dataloader

import torchvision.transforms as T

def GetTransform(par, type):

    if type == 'train':
        transforms = T.Compose([
            T.Resize((256, 256)),
            T.RandomRotation(30),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.RandomResizedCrop(224, scale=(0.8, 1.0)),
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3)
        ])
    else:
        transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3)
        ])
    return transforms


if __name__ == '__main__':
    #change dir to root dir
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(root_dir)

    from experiments.configs.par import Struct
    import yaml

    #load par
    par_dir = './experiments/configs/Classification.yaml'
    par = yaml.safe_load(open(par_dir,'r'))
    par = Struct(**par)
    transform = GetTransform(par, 'train')
    dataset, loader = GetDataset(par, 'train',transform)

    for img, label in loader:
        print(len(label))

