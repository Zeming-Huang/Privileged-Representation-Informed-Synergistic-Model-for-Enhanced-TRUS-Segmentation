# PRISM

Official implementation of PRISM.

## Overview

This release contains the core dual-modal training and evaluation path used for PRISM:

- `train_dual_modal.py` for training
- `evaluate.py` for evaluation

The public release is intentionally narrow and focuses on the main training and evaluation workflow.

## Expected Data Layout

```text
data/
  train/
    trus/
      imgs/
      gts/
    mri/
      imgs/
      gts/
  val/
    trus/
      imgs/
      gts/
    mri/
      imgs/
      gts/
```

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python train_dual_modal.py \
  -trus_data_root ./data/train/trus \
  -mri_data_root ./data/train/mri \
  -val_trus_data_root ./data/val/trus \
  -val_mri_data_root ./data/val/mri \
  -pretrained_checkpoint ./checkpoints/lite_medsam.pth \
  -work_dir ./work_dir/prism
```

## Evaluation

```bash
python evaluate.py \
  -pred_dir ./predictions \
  -gt_dir ./ground_truth \
  -output_csv ./evaluation_results.csv
```

## Notes

- `-trus_pretrained_checkpoint` and `-mri_pretrained_checkpoint` are optional. If omitted, the script falls back to `-pretrained_checkpoint`.
- The evaluation script expects prediction `.npz` files with a `segs` array and ground-truth `.npz` files with a `gts` array.
