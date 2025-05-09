#!/bin/bash

# Script to run the full SimCLR training and evaluation pipeline

# Default parameters
DATA_DIR=""
BATCH_SIZE_PRETRAIN=256
BATCH_SIZE_TRAIN=128
BATCH_SIZE_EVAL=32
EPOCHS_PRETRAIN=200
EPOCHS_TRAIN=50
LR_PRETRAIN=0.0003
LR_TRAIN=0.0001
TEMP=0.5
PROJ_DIM=128
BACKBONE="resnet50"
NUM_WORKERS=8
FREEZE_ENCODER=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --batch-size-pretrain)
      BATCH_SIZE_PRETRAIN="$2"
      shift 2
      ;;
    --batch-size-train)
      BATCH_SIZE_TRAIN="$2"
      shift 2
      ;;
    --batch-size-eval)
      BATCH_SIZE_EVAL="$2"
      shift 2
      ;;
    --epochs-pretrain)
      EPOCHS_PRETRAIN="$2"
      shift 2
      ;;
    --epochs-train)
      EPOCHS_TRAIN="$2"
      shift 2
      ;;
    --lr-pretrain)
      LR_PRETRAIN="$2"
      shift 2
      ;;
    --lr-train)
      LR_TRAIN="$2"
      shift 2
      ;;
    --temperature)
      TEMP="$2"
      shift 2
      ;;
    --projection-dim)
      PROJ_DIM="$2"
      shift 2
      ;;
    --backbone)
      BACKBONE="$2"
      shift 2
      ;;
    --workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --no-freeze)
      FREEZE_ENCODER=false
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Validate required parameters
if [ -z "$DATA_DIR" ]; then
  echo "Error: --data-dir is required"
  exit 1
fi

# Create directories
mkdir -p checkpoints/simclr_pretrain
mkdir -p checkpoints/classifier
mkdir -p runs/simclr_pretrain
mkdir -p runs/classifier
mkdir -p results

echo "==============================================="
echo "Starting SimCLR pipeline with configuration:"
echo "==============================================="
echo "Data directory: $DATA_DIR"
echo "Batch size (pretrain): $BATCH_SIZE_PRETRAIN"
echo "Batch size (train): $BATCH_SIZE_TRAIN"
echo "Batch size (eval): $BATCH_SIZE_EVAL"
echo "Epochs (pretrain): $EPOCHS_PRETRAIN"
echo "Epochs (train): $EPOCHS_TRAIN"
echo "Learning rate (pretrain): $LR_PRETRAIN"
echo "Learning rate (train): $LR_TRAIN"
echo "Temperature: $TEMP"
echo "Projection dimension: $PROJ_DIM"
echo "Backbone: $BACKBONE"
echo "Number of workers: $NUM_WORKERS"
echo "Freeze encoder: $FREEZE_ENCODER"
echo "==============================================="

# Step 1: Pre-train SimCLR model
echo "Step 1: Pre-training SimCLR model..."
python train.py \
  --data-dir "$DATA_DIR" \
  --batch-size "$BATCH_SIZE_PRETRAIN" \
  --epochs "$EPOCHS_PRETRAIN" \
  --lr "$LR_PRETRAIN" \
  --temperature "$TEMP" \
  --projection-dim "$PROJ_DIM" \
  --backbone "$BACKBONE" \
  --num-workers "$NUM_WORKERS" \
  --mode pretrain \
  --log-dir runs/simclr_pretrain \
  --checkpoint-dir checkpoints/simclr_pretrain

# Check if pretraining was successful
if [ ! -f "checkpoints/simclr_pretrain/checkpoint_${EPOCHS_PRETRAIN}.pth" ]; then
  if [ ! -f "checkpoints/simclr_pretrain/model_best.pth" ]; then
    echo "Error: Pre-training did not complete successfully."
    exit 1
  fi
fi

# Step 2: Train classifier on top of SimCLR
echo "Step 2: Training classifier on top of SimCLR..."
FREEZE_FLAG=""
if [ "$FREEZE_ENCODER" = true ]; then
  FREEZE_FLAG="--freeze-encoder"
fi

python train.py \
  --data-dir "$DATA_DIR" \
  --batch-size "$BATCH_SIZE_TRAIN" \
  --epochs "$EPOCHS_TRAIN" \
  --lr "$LR_TRAIN" \
  --projection-dim "$PROJ_DIM" \
  --backbone "$BACKBONE" \
  --num-workers "$NUM_WORKERS" \
  --mode train \
  --pretrained checkpoints/simclr_pretrain/model_best.pth \
  --log-dir runs/classifier \
  --checkpoint-dir checkpoints/classifier \
  $FREEZE_FLAG

# Check if classifier training was successful
if [ ! -f "checkpoints/classifier/classifier_${EPOCHS_TRAIN}.pth" ]; then
  if [ ! -f "checkpoints/classifier/model_best.pth" ]; then
    echo "Error: Classifier training did not complete successfully."
    exit 1
  fi
fi

# Step 3: Evaluate classifier
echo "Step 3: Evaluating classifier..."
python train.py \
  --data-dir "$DATA_DIR" \
  --batch-size "$BATCH_SIZE_EVAL" \
  --mode eval \
  --pretrained checkpoints/classifier/model_best.pth \
  --num-workers "$NUM_WORKERS" \
  --backbone "$BACKBONE"

# Step 4: Run benchmark
echo "Step 4: Running benchmarks..."
python benchmark.py \
  --data-dir "$DATA_DIR" \
  --batch-size "$BATCH_SIZE_EVAL" \
  --model-path checkpoints/classifier/model_best.pth \
  --num-workers "$NUM_WORKERS" \
  --visualize

echo "==============================================="
echo "SimCLR pipeline complete!"
echo "Results are saved in the results directory."
echo "===============================================" 