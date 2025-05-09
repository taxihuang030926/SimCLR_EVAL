import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import json

class ImageNetCustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.categories = {
            'Animals': 0,
            'Vehicles': 1,
            'Plants': 2,
            'Buildings and Structures': 3,
            'Clothing and Accessories': 4
        }
        
        # ImageNet category mapping (simplified version)
        self.imagenet_to_custom = {
            # Animals
            'n01440764': 0, 'n01443537': 0, 'n01484850': 0, 'n01491361': 0,
            # Vehicles
            'n02690373': 1, 'n02691156': 1, 'n02692877': 1, 'n02701002': 1,
            # Plants
            'n11939491': 2, 'n11950345': 2, 'n11879895': 2, 'n12057211': 2,
            # Buildings and Structures
            'n02814533': 3, 'n02787622': 3, 'n02788148': 3, 'n02894337': 3,
            # Clothing and Accessories
            'n03124043': 4, 'n03125729': 4, 'n03131574': 4, 'n03141823': 4
        }
        
        self.samples = self._load_samples()
        
    def _load_samples(self):
        samples = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.JPEG'):
                    path = os.path.join(root, file)
                    synset = os.path.basename(os.path.dirname(path))
                    if synset in self.imagenet_to_custom:
                        label = self.imagenet_to_custom[synset]
                        samples.append((path, label))
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    
    return train_transform, val_transform

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res 