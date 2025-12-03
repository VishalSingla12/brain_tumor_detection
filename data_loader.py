import os
import shutil
import kagglehub
import random
from pathlib import Path
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import config
import utils

def download_dataset():
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
    print(f"Dataset downloaded to: {path}")
    return path

def organize_dataset(source_path):
    print("Organizing dataset...")
    source_path = Path(source_path)
    training_dirs = list(source_path.rglob('Training'))
    
    if not training_dirs:
        pass
    else:
        source_train = training_dirs[0]
        source_test = source_train.parent / 'Testing'
    for split in ['train', 'val', 'test']:
        for class_name in config.CLASS_NAMES:
            os.makedirs(os.path.join(config.DATA_DIR, split, class_name), exist_ok=True)
    all_files = {class_name: [] for class_name in config.CLASS_NAMES}
    def scan_and_collect(root_dir):
        if not root_dir.exists(): return
        for class_name in config.CLASS_NAMES:
            class_dir = root_dir / class_name
            if not class_dir.exists():
                found = False
                for child in root_dir.iterdir():
                    if child.is_dir() and child.name.lower() == class_name.lower():
                        class_dir = child
                        found = True
                        break
                if not found: continue
            
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    all_files[class_name].append(img_path)

    scan_and_collect(source_train)
    if source_test.exists():
        scan_and_collect(source_test)
    utils.seed_everything()
    
    for class_name, files in all_files.items():
        random.shuffle(files)
        n = len(files)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)
        train_files = files[:n_train]
        val_files = files[n_train:n_train+n_val]
        test_files = files[n_train+n_val:]
        print(f"Class {class_name}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
        for f in train_files:
            shutil.copy(f, os.path.join(config.TRAIN_DIR, class_name, f.name))
        for f in val_files:
            shutil.copy(f, os.path.join(config.VAL_DIR, class_name, f.name))
        for f in test_files:
            shutil.copy(f, os.path.join(config.TEST_DIR, class_name, f.name))

def get_transforms():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    val_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return train_transform, val_transform

def create_dataloaders():
    train_transform, val_transform = get_transforms()
    if not os.path.exists(config.TRAIN_DIR):
        raw_path = download_dataset()
        organize_dataset(raw_path)
    train_dataset = datasets.ImageFolder(config.TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(config.VAL_DIR, transform=val_transform)
    test_dataset = datasets.ImageFolder(config.TEST_DIR, transform=val_transform)
    targets = train_dataset.targets
    class_counts = np.bincount(targets)
    class_weights = 1. / class_counts
    sample_weights = class_weights[targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, train_dataset.classes

if __name__ == "__main__":
    utils.seed_everything()
    if os.path.exists(config.DATA_DIR):
        print(f"Data directory {config.DATA_DIR} already exists. Skipping download/organize.")
    else:
        raw_path = download_dataset()
        organize_dataset(raw_path)
    
    tr, va, te, classes = create_dataloaders()
    print(f"DataLoaders created. Classes: {classes}")
    print(f"Train batches: {len(tr)}")
    print(f"Val batches: {len(va)}")
    print(f"Test batches: {len(te)}")
