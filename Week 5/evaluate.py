"""
evaluate.py - Standalone Evaluation and Model Inference Verification
Week 5 Task: Deep Learning Application in Data Science
"""

import argparse
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix
from model import CustomResNetCIFAR

CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Trained Model on CIFAR-10 Test Set")
    parser.add_argument("--checkpoint", type=str, default="best_model.pth", help="Trained model weights path")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory for CIFAR-10 data")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = CustomResNetCIFAR(num_classes=10, dropout_rate=0.0, use_se=True).to(device)
    
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded weights from {args.checkpoint}")
    else:
        print(f"Warning: Checkpoint '{args.checkpoint}' not found. Using initialized weights for dry-run.")

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = (all_preds == all_targets).mean()
    print(f"\nTest Set Top-1 Accuracy: {acc * 100:.2f}%\n")

    print("Detailed Classification Report:")
    print(classification_report(all_targets, all_preds, target_names=CLASSES, digits=4))

    print("Confusion Matrix:")
    cm = confusion_matrix(all_targets, all_preds)
    print(cm)

if __name__ == "__main__":
    main()
