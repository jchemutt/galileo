# Galileo Pretrained Remote Sensing Model – Rwanda Crop Type Classification

A fine-tuning pipeline for the **[Galileo Pretrained Remote Sensing Model](https://github.com/nasaharvest/galileo)** to classify crop types using pixel-wise segmentation with multi-temporal Sentinel-1 and Sentinel-2 data in Rwanda.

---

## Quickstart

### 1. Clone the repository

```bash
git clone <repo-url>
cd galileo
```

### 2. Create Environment

With **conda**:

```bash
conda env create -f environment.yml
conda activate galileo-env
```

## Dataset Preparation

### Step 1: Export Sentinel-1/2 Data from Earth Engine

Use the script below to export Sentinel-1/ Sentinel-2 images over your area of interest for specific session (Season A or Season B):

```bash
python run_export_rwanda.py
```

### Step 2: Download Exported Tiles from GEE

After GEE completes your export tasks, download the resulting GeoTIFF tiles::

```bash
python download_tiles_from_gee.py
```

### Step 3: Extract Pixel-wise Patches from Tiles

Use preprocessed Sentinel-1/Sentinel-2 tiles and polygon labels to generate 8×8 pixel-wise training patches:

```bash
python extract_patches_from_tiles.py
```

This script will:

- Read raster tiles and label masks
- Extract valid 8×8 image and mask patches
- Save `.npz` patch datasets for `train`, `val`, and `test` splits

### Folder Structure Example

```
data/
  patches/
    train/
      inputs/
      masks/
    val/
    test/
```

---

## Fine-Tuning

Train the pixel-wise classifier using the Galileo encoder:

```bash
python finetune_rwanda.py \
  --data_dir data/patches/ \
  --encoder_ckpt data/models/nano/ \
  --save_dir checkpoints/ \
  --batch_size 8 \
  --epochs 50 \
  --lr 0.0001
```

## Inference

Run inference on prepared raster tiles:

```bash
python inference.py
```

This script will:

- Load each tile (60-band: 5 timesteps × 12 bands)
- Apply a sliding window to generate 8×8 predictions
- Merge predictions and output a GeoTIFF (`merged_prediction.tif`)

### Important:

- Your input tiles must have **60 bands** (5 months × 12 bands).
- Invalid or low-confidence pixels are masked using a configurable `confidence_threshold`.

---
