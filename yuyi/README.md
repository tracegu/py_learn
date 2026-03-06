# Semantic Segmentation Training (yuyi)

This directory contains a simple PyTorch-based semantic segmentation
training pipeline. It assumes you have a set of input images and a JSON
file with segmentation annotations in a COCO-like format.

## Files

* `dataset.py` - dataset class that rasterizes polygon annotations into
  per-pixel masks.
* `model.py` - a wrapper around torchvision's DeepLabV3 or a basic U-Net
  implementation.
* `train.py` - training script that loads the dataset, trains on GPU if
  available, and saves the model state.

## Usage

```bash
python train.py \
    --images /path/to/images \
    --annotations /path/to/annotations.json \
    --num-classes 21 \
    --epochs 50 \
    --batch-size 8 \
    --output ./seg_model.pth
```

You can specify `--backbone unet` to use the internal UNet instead of
DeepLabV3. The script will automatically select a CUDA device if one is
available.
