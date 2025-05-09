import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import argparse
import time
import numpy as np
from tqdm import tqdm

from model import SimCLR, NT_Xent
from utils import ImageNetCustomDataset, get_transforms, accuracy

def validate(val_loader, model, criterion, device):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    contrastive_losses = AverageMeter('Contrastive Loss', ':.4e')
    class_losses = AverageMeter('Classification Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    # switch to evaluate mode
    model.eval()
    
    class_criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)
            
            # compute output
            proj, logits = model(images)
            
            # compute contrastive loss
            contrastive_loss = criterion(proj, proj)
            
            # compute classification loss
            class_loss = class_criterion(logits, target)
            
            # combined loss
            loss = contrastive_loss + class_loss
            
            # measure accuracy and record loss
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            contrastive_losses.update(contrastive_loss.item(), images.size(0))
            class_losses.update(class_loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % 10 == 0:
                print(f'Test: [{i}/{len(val_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})')
    
    print(f' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')
    return {
        'acc1': top1.avg,
        'acc5': top5.avg,
        'loss': losses.avg,
        'contrastive_loss': contrastive_losses.avg,
        'classification_loss': class_losses.avg
    }

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def main():
    parser = argparse.ArgumentParser(description='SimCLR Validation')
    parser.add_argument('--data', metavar='DIR', help='path to ImageNet dataset')
    parser.add_argument('--batch-size', default=256, type=int, help='batch size')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
    parser.add_argument('--checkpoint', default='', type=str, help='path to checkpoint')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # create model
    print("Creating model")
    model = SimCLR()
    
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
    
    model = model.to(device)
    
    # define loss function
    criterion = NT_Xent()
    
    # Data loading
    _, val_transform = get_transforms()
    
    val_dataset = ImageNetCustomDataset(
        args.data,
        transform=val_transform
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    
    # validate
    results = validate(val_loader, model, criterion, device)
    
    print("\nValidation Results:")
    print(f"Top-1 Accuracy: {results['acc1']:.2f}%")
    print(f"Top-5 Accuracy: {results['acc5']:.2f}%")
    print(f"Total Loss: {results['loss']:.4f}")
    print(f"Contrastive Loss: {results['contrastive_loss']:.4f}")
    print(f"Classification Loss: {results['classification_loss']:.4f}")

if __name__ == '__main__':
    main() 