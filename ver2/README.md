# SimCLR for Image Classification

This project implements SimCLR (Simple Framework for Contrastive Learning of Representations) using PyTorch, trained on ImageNet and fine-tuned for a 5-category classification task:

- Animals
- Vehicles
- Plants
- Buildings and Structures
- Clothing and Accessories

## Overview

SimCLR is a self-supervised learning framework that learns image representations via contrastive learning. The implementation consists of:

1. A base encoder (ResNet50) that extracts feature representations
2. A projection head that maps representations to a space where contrastive loss is applied
3. A data augmentation module that creates different views of the same image
4. NT-Xent (Normalized Temperature-scaled Cross Entropy) loss function for contrastive learning

After pre-training, the model is fine-tuned for the classification task.

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Project Structure

- `model.py`: SimCLR model implementation
- `loss.py`: NT-Xent loss implementation
- `dataset.py`: Data loading and augmentation for contrastive learning
- `train.py`: Training and evaluation scripts
- `utils.py`: Utility functions for tracking metrics and visualization
- `benchmark.py`: Benchmark script for model evaluation

## Usage

### 1. Pre-train SimCLR

```bash
python train.py \
    --data-dir /path/to/imagenet \
    --batch-size 256 \
    --epochs 200 \
    --lr 0.0003 \
    --temperature 0.5 \
    --projection-dim 128 \
    --mode pretrain \
    --log-dir runs/simclr_pretrain \
    --checkpoint-dir checkpoints/simclr_pretrain
```

### 2. Train Classifier

```bash
python train.py \
    --data-dir /path/to/imagenet \
    --batch-size 128 \
    --epochs 50 \
    --lr 0.0001 \
    --mode train \
    --pretrained checkpoints/simclr_pretrain/model_best.pth \
    --log-dir runs/classifier \
    --checkpoint-dir checkpoints/classifier \
    --freeze-encoder
```

### 3. Evaluate Model

```bash
python train.py \
    --data-dir /path/to/imagenet \
    --batch-size 128 \
    --mode eval \
    --pretrained checkpoints/classifier/model_best.pth
```

### 4. Benchmark

```bash
python benchmark.py \
    --data-dir /path/to/imagenet \
    --batch-size 32 \
    --model-path checkpoints/classifier/model_best.pth \
    --visualize
```

## Results

The benchmark will generate:
- Classification report with precision, recall, and F1-score
- Confusion matrix visualization
- t-SNE visualization of feature embeddings (if `--visualize` flag is set)
- Inference time metrics

## References

- SimCLR Paper: [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
- ImageNet Dataset: [ImageNet](http://image-net.org/) 