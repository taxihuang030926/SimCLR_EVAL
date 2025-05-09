import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from model import SimCLR, SimCLRClassifier
from loss import NTXentLoss
from dataset import get_dataloaders
from utils import AverageMeter, accuracy

def train_simclr(args):
    """
    Train SimCLR model using contrastive learning
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = SimCLR(base_encoder=args.backbone, projection_dim=args.projection_dim).to(device)
    
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Initialize criterion
    criterion = NTXentLoss(batch_size=args.batch_size, temperature=args.temperature).to(device)
    
    # Initialize dataloaders
    train_loader, _ = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        contrastive=True
    )
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Training loop
    model.train()
    for epoch in range(args.start_epoch, args.epochs):
        losses = AverageMeter('Loss', ':.4e')
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        
        for idx, (x_i, x_j) in enumerate(pbar):
            optimizer.zero_grad()
            
            x_i = x_i.to(device)
            x_j = x_j.to(device)
            
            # Get projections
            _, z_i = model(x_i)
            _, z_j = model(x_j)
            
            # Calculate loss
            loss = criterion(z_i, z_j)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            # Update metrics
            losses.update(loss.item(), args.batch_size)
            pbar.set_postfix({'loss': losses.avg})
            
        # Log to tensorboard
        writer.add_scalar('train/loss', losses.avg, epoch)
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs:
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': losses.avg,
            }, is_best=False, filename=os.path.join(args.checkpoint_dir, f'checkpoint_{epoch+1}.pth'))
    
    return model

def train_classifier(args, pretrained_model=None):
    """
    Train the classifier on top of the SimCLR model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    if pretrained_model is None:
        # Load from checkpoint
        model = SimCLR(base_encoder=args.backbone, projection_dim=args.projection_dim).to(device)
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = pretrained_model
    
    # Create classifier
    classifier = SimCLRClassifier(
        base_encoder=args.backbone, 
        num_classes=5, 
        projection_dim=args.projection_dim
    ).to(device)
    
    # Initialize encoder with pretrained weights
    classifier.simclr.load_state_dict(model.state_dict())
    
    # Freeze encoder parameters
    if args.freeze_encoder:
        for param in classifier.simclr.encoder.parameters():
            param.requires_grad = False
    
    # Initialize optimizer
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, classifier.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Initialize criterion
    criterion = nn.CrossEntropyLoss().to(device)
    
    # Initialize dataloaders
    train_loader, val_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        contrastive=False
    )
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Training loop
    best_acc = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        # Train
        train_loss, train_acc = train_epoch(
            classifier, train_loader, criterion, optimizer, epoch, device, args
        )
        
        # Validate
        val_loss, val_acc = validate(
            classifier, val_loader, criterion, device
        )
        
        # Log to tensorboard
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/acc', train_acc, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/acc', val_acc, epoch)
        
        # Save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        
        if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs or is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, is_best, filename=os.path.join(args.checkpoint_dir, f'classifier_{epoch+1}.pth'))
    
    return classifier

def train_epoch(model, train_loader, criterion, optimizer, epoch, device, args):
    """
    Train for one epoch
    """
    model.train()
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
    
    for idx, (images, target) in enumerate(pbar):
        images = images.to(device)
        target = target.to(device)
        
        # Forward
        output = model(images)
        loss = criterion(output, target)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        acc1, = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        
        pbar.set_postfix({'loss': losses.avg, 'acc': top1.avg})
    
    return losses.avg, top1.avg

def validate(model, val_loader, criterion, device):
    """
    Validate the model
    """
    model.eval()
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    
    with torch.no_grad():
        for images, target in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            target = target.to(device)
            
            # Forward
            output = model(images)
            loss = criterion(output, target)
            
            # Update metrics
            acc1, = accuracy(output, target, topk=(1,))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
    
    print(f'Validation Loss: {losses.avg:.4f}, Accuracy: {top1.avg:.2f}%')
    
    return losses.avg, top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    """
    Save checkpoint
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(os.path.dirname(filename), 'model_best.pth')
        torch.save(state, best_filename)

def main():
    parser = argparse.ArgumentParser(description='PyTorch SimCLR Training')
    parser.add_argument('--data-dir', type=str, required=True, help='path to ImageNet dataset')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('--epochs', type=int, default=200, help='number of total epochs to run')
    parser.add_argument('--start-epoch', type=int, default=0, help='manual epoch number (useful on restarts)')
    parser.add_argument('--lr', '--learning-rate', dest='learning_rate', type=float, default=0.0003, help='initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--temperature', type=float, default=0.5, help='softmax temperature')
    parser.add_argument('--projection-dim', type=int, default=128, help='projection dimension')
    parser.add_argument('--backbone', type=str, default='resnet50', help='encoder backbone')
    parser.add_argument('--num-workers', type=int, default=8, help='number of workers')
    parser.add_argument('--log-dir', type=str, default='runs', help='path to tensorboard logs')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='path to save checkpoints')
    parser.add_argument('--save-freq', type=int, default=10, help='save frequency')
    parser.add_argument('--resume', type=str, default='', help='path to checkpoint')
    parser.add_argument('--pretrained', type=str, default='', help='path to pretrained model')
    parser.add_argument('--mode', type=str, choices=['pretrain', 'train', 'eval'], required=True, help='training mode')
    parser.add_argument('--freeze-encoder', action='store_true', help='freeze encoder parameters')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    if args.mode == 'pretrain':
        # Train SimCLR model
        print("Starting SimCLR pretraining...")
        model = train_simclr(args)
    elif args.mode == 'train':
        # Train classifier
        print("Starting classifier training...")
        train_classifier(args)
    elif args.mode == 'eval':
        # Evaluate
        print("Evaluating model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SimCLRClassifier(base_encoder=args.backbone, num_classes=5).to(device)
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        _, val_loader = get_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        criterion = nn.CrossEntropyLoss().to(device)
        _, acc = validate(model, val_loader, criterion, device)
        print(f"Final accuracy: {acc:.2f}%")

if __name__ == '__main__':
    main() 