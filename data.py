import random

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config import BATCH_SIZE, SEED, VAL_SPLIT


def build_transforms():
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    return train_tf, val_tf


def _split_indices(total, val_split, seed):
    indices = list(range(total))
    random.seed(seed)
    random.shuffle(indices)
    val_size = int(total * val_split)
    return indices[val_size:], indices[:val_size]


def load_datasets(data_dir, train_tf, val_tf):
    train_base = datasets.ImageFolder(data_dir, transform=train_tf)
    val_base = datasets.ImageFolder(data_dir, transform=val_tf)
    train_indices, val_indices = _split_indices(len(train_base), VAL_SPLIT, SEED)
    return Subset(train_base, train_indices), Subset(val_base, val_indices), train_base.classes


def build_dataloaders(train_set, val_set):
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader
