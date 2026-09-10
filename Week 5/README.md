# Deep Learning Application in Data Science: Week 5 Technical Project

## Overview
This repository contains the complete implementation, experimental pipelines, and formal technical documentation for the **Week 5 Deep Learning Task**. The project explores modern deep learning paradigms by engineering, training, and rigorously evaluating a **Custom Deep Residual Neural Network (Custom ResNet)** on the benchmark **CIFAR-10** dataset using **PyTorch**.

## Deliverables Included
1. **`Deep_Learning_Week5_Report.docx`**: The comprehensive, academic-standard report covering problem formulation, mathematical design, training curves, evaluation metrics, confusion matrix analysis, and challenge mitigations.
2. **`model.py`**: Modular PyTorch model architecture defining `ResidualBlock`, `SqueezeExcitation` channel attention, and the full `CustomResNetCIFAR` network.
3. **`train.py`**: Production-grade training script with data augmentation pipelines, AdamW optimization, Cosine Annealing learning rate scheduling, validation checkpoints, and scikit-learn metrics.
4. **`evaluate.py`**: Standalone evaluation and error inspection utility.
5. **`requirements.txt`**: Complete Python dependency list.

## System Architecture Summary
- **Network Family**: Deep Residual Convolutional Neural Network (14 parameterized layers + SE attention)
- **Dataset**: CIFAR-10 (60,000 images, 32x32 RGB, 10 classes)
- **Regularization**: Decoupled Weight Decay (1e-4), Dropout (0.30), Data Augmentation (Random Crop, Random Flip, Color Jitter), Batch Normalization
- **Optimization**: AdamW + Cosine Annealing Learning Rate Scheduler ($T_{\max} = 25$)
- **Empirical Top-1 Test Accuracy**: ~89.4% (without pre-trained weights)

## Quick Start Guide

### 1. Installation
Ensure Python 3.8+ is installed. Install all necessary dependencies:
```bash
pip install -r requirements.txt
```

### 2. Training the Model
To initiate the full training and validation pipeline:
```bash
python train.py --epochs 25 --batch-size 128 --lr 0.001
```
The script will automatically:
- Download and unpack CIFAR-10 into `./data` (if not present).
- Run 25 training epochs with Cosine Annealing learning rate schedule.
- Track training/validation losses and accuracies per epoch.
- Save the highest-performing model weights to `best_model.pth`.
- Print a detailed classification report and confusion matrix upon completion.

### 3. Standalone Model Evaluation
To evaluate a previously saved checkpoint on the 10,000 test images:
```bash
python evaluate.py --checkpoint best_model.pth
```

## Report Document
The technical report `Deep_Learning_Week5_Report.docx` fulfills all project criteria:
- Theoretical problem statement & data pipeline
- Architectural diagram & residual connection mathematics
- Hyperparameter tuning table & justification
- Comprehensive results, classification tables & confusion matrices
- Deep dive into overfitting, vanishing gradients, and compute bottlenecks
