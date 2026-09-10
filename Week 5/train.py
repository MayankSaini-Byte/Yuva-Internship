"""
train.py - Full Training and Validation Pipeline for CIFAR-10 ResNet
Week 5 Task: Deep Learning Application in Data Science
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from model import CustomResNetCIFAR

CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

def parse_args():
    parser = argparse.ArgumentParser(description="Train Custom ResNet on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training and eval")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 regularization factor (AdamW)")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability before classifier")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory to store CIFAR-10")
    parser.add_argument("--save-path", type=str, default="best_model.pth", help="Path to save best checkpoint")
    return parser.parse_args()

def get_dataloaders(data_dir: str, batch_size: int):
    # Standard CIFAR-10 per-channel mean and standard deviation
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, test_loader

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    return avg_loss, acc, np.array(all_preds), np.array(all_targets)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"===========================================================")
    print(f" Deep Learning CIFAR-10 Training Pipeline (Week 5 Task)     ")
    print(f" Compute Device        : {device}")
    print(f" Number of Epochs      : {args.epochs}")
    print(f" Batch Size            : {args.batch_size}")
    print(f" Initial Learning Rate : {args.lr}")
    print(f" Weight Decay (AdamW)  : {args.weight_decay}")
    print(f" Dropout Rate          : {args.dropout}")
    print(f"===========================================================\n")

    train_loader, test_loader = get_dataloaders(args.data_dir, args.batch_size)
    model = CustomResNetCIFAR(num_classes=10, dropout_rate=args.dropout, use_se=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_acc = 0.0
    training_history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        epoch_time = time.time() - t0

        curr_lr = optimizer.param_groups[0]['lr']
        training_history.append((epoch, train_loss, train_acc, val_loss, val_acc, curr_lr))

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:6.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:6.2f}% | "
              f"LR: {curr_lr:.6f} | ({epoch_time:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_acc,
            }, args.save_path)
            print(f"  --> Checkpoint saved with improved validation accuracy: {val_acc*100:.2f}%")

    print(f"\n===========================================================")
    print(f" Training Complete! Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f" Loading best checkpoint from {args.save_path} for final evaluation...")
    print(f"===========================================================\n")

    checkpoint = torch.load(args.save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, preds, targets = evaluate(model, test_loader, criterion, device)
    print(f"Final Test Accuracy : {test_acc*100:.2f}%")
    print(f"Final Test Loss     : {test_loss:.4f}\n")

    print("Detailed Classification Report:")
    print(classification_report(targets, preds, target_names=CLASSES, digits=4))

    print("Confusion Matrix:")
    cm = confusion_matrix(targets, preds)
    print(cm)

if __name__ == "__main__":
    main()
