# SimCLR ImageNet Validation with Custom Categories

This implementation provides a validation script for SimCLR model on ImageNet dataset with custom category grouping. The categories are:
- Animals
- Vehicles
- Plants
- Buildings and Structures
- Clothing and Accessories

## Requirements

Install the required packages:
```bash
pip install -r requirements.txt
```

## Dataset Structure

The ImageNet dataset should be organized in the standard ImageNet structure:
```
imagenet/
    n01440764/
        images...
    n01443537/
        images...
    ...
```

## Usage

To validate a pre-trained SimCLR model:

```bash
python validate.py --data /path/to/imagenet \
                  --batch-size 256 \
                  --workers 4 \
                  --checkpoint /path/to/checkpoint.pth
```

Arguments:
- `--data`: Path to ImageNet dataset directory
- `--batch-size`: Batch size for validation (default: 256)
- `--workers`: Number of data loading workers (default: 4)
- `--checkpoint`: Path to the model checkpoint file

## Output

The validation script will output:
- Top-1 and Top-5 accuracy
- Total loss (combination of contrastive and classification losses)
- Contrastive loss
- Classification loss

## Model Architecture

The implementation uses:
- ResNet50 backbone (pre-trained on ImageNet)
- Projection head for contrastive learning
- Classification head for the 5 custom categories
- NT-Xent (Normalized Temperature-scaled Cross Entropy) loss for contrastive learning

## Notes

- The model performs both contrastive learning validation and classification into the 5 custom categories
- The validation process measures both the quality of the learned representations (through contrastive loss) and the classification performance
- The implementation uses the SimCLR architecture with modifications for custom category classification 