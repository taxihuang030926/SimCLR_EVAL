import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import numpy as np

# Define categories and class mapping
CATEGORIES = {
    'Animals': 0,
    'Vehicles': 1,
    'Plants': 2,
    'Buildings and Structures': 3,
    'Clothing and Accessories': 4
}

# Mapping from ImageNet classes to our 5 categories
# This is a simplified mapping - in a real implementation, we would map all 1000 ImageNet classes
IMAGENET_TO_CATEGORY = {
    # Animals (mammals, birds, fish, reptiles, etc.)
    'n01440764': 0,  # tench
    'n01443537': 0,  # goldfish
    'n01484850': 0,  # great white shark
    'n01608432': 0,  # kite (bird)
    'n01820546': 0,  # lizard
    'n01910747': 0,  # jellyfish
    
    # Vehicles
    'n02701002': 1,  # ambulance
    'n02814533': 1,  # beach wagon
    'n02930766': 1,  # cab
    'n03100240': 1,  # convertible
    'n03594945': 1,  # jeep
    'n04285008': 1,  # sports car
    'n04465501': 1,  # train
    
    # Plants
    'n11939491': 2,  # daisy
    'n12267677': 2,  # acorn
    'n13054560': 2,  # bolete (mushroom)
    'n13133613': 2,  # ear of corn
    
    # Buildings and Structures
    'n02980441': 3,  # castle
    'n03026506': 3,  # church
    'n03028079': 3,  # church tower
    'n03788195': 3,  # mosque
    'n04346328': 3,  # stupa
    
    # Clothing and Accessories
    'n02883205': 4,  # bow tie
    'n03124170': 4,  # cowboy hat
    'n03527444': 4,  # handbag
    'n03877472': 4,  # pajama
    'n04209133': 4   # shower cap
}

class SimCLRDataAugmentation:
    def __init__(self, size=224):
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=int(0.1 * size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.eval_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)

class SimCLRDataset(Dataset):
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, _ = self.base_dataset[idx]
        
        if self.transform:
            img1, img2 = self.transform(img)
            return img1, img2
        
        return img, img

class ClassificationDataset(Dataset):
    def __init__(self, base_dataset, transform=None, target_transform=None):
        self.base_dataset = base_dataset
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, target = self.base_dataset[idx]
        
        # Map original ImageNet target to our 5 categories
        synset = self.base_dataset.classes[target]
        if synset in IMAGENET_TO_CATEGORY:
            new_target = IMAGENET_TO_CATEGORY[synset]
        else:
            # Default to a category or skip this sample
            new_target = 0  # Default to Animals
        
        if self.transform:
            img = self.transform(img)
        
        if self.target_transform:
            new_target = self.target_transform(new_target)
        
        return img, new_target

def get_imagenet_subset(data_dir, transform=None, train=True, num_samples=1000):
    """
    Create a subset of ImageNet for our 5 categories
    """
    dataset = torchvision.datasets.ImageNet(
        root=data_dir,
        split='train' if train else 'val',
        transform=transform
    )
    
    # Filter samples to only include our categories
    indices = []
    for idx, (_, target) in enumerate(dataset.samples):
        synset = dataset.classes[target]
        if synset in IMAGENET_TO_CATEGORY:
            indices.append(idx)
    
    # Limit number of samples
    if num_samples and num_samples < len(indices):
        indices = indices[:num_samples]
    
    return Subset(dataset, indices)

def get_dataloaders(data_dir, batch_size=128, num_workers=8, contrastive=False):
    """
    Get dataloaders for contrastive learning and classification
    """
    data_augmentation = SimCLRDataAugmentation()
    
    # Training data
    if contrastive:
        train_dataset = get_imagenet_subset(data_dir, transform=None, train=True)
        train_dataset = SimCLRDataset(train_dataset, transform=data_augmentation)
    else:
        train_dataset = get_imagenet_subset(data_dir, transform=data_augmentation.train_transform, train=True)
        train_dataset = ClassificationDataset(train_dataset)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Validation data
    val_dataset = get_imagenet_subset(data_dir, transform=data_augmentation.eval_transform, train=False)
    val_dataset = ClassificationDataset(val_dataset)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader 