#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive evaluation for PRISM prediction `.npz` files.

Expected input format:
- prediction files contain `segs` and optionally `spacing`
- ground-truth files contain `gts`
"""

import argparse
import os
from glob import glob
from os.path import basename, join

import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, distance_transform_edt
from tqdm import tqdm


def compute_dice_score(pred, gt):
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt)
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2.0 * intersection / union


def compute_iou(pred, gt):
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersection
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def compute_sensitivity(pred, gt):
    tp = np.sum(pred * gt)
    fn = np.sum((1 - pred) * gt)
    if tp + fn == 0:
        return 1.0
    return tp / (tp + fn)


def compute_specificity(pred, gt):
    tn = np.sum((1 - pred) * (1 - gt))
    fp = np.sum(pred * (1 - gt))
    if tn + fp == 0:
        return 1.0
    return tn / (tn + fp)


def compute_precision(pred, gt):
    tp = np.sum(pred * gt)
    fp = np.sum(pred * (1 - gt))
    if tp + fp == 0:
        return 1.0
    return tp / (tp + fp)


def normalize_spacing(spacing):
    if isinstance(spacing, np.ndarray):
        spacing = spacing.flatten()
    else:
        spacing = np.array([spacing] if np.isscalar(spacing) else spacing)

    if len(spacing) == 0:
        return np.array([1.0, 1.0, 1.0], dtype=np.float32)
    if len(spacing) == 1:
        return np.array([spacing[0], spacing[0], 1.0], dtype=np.float32)
    if len(spacing) == 2:
        return np.array([spacing[0], spacing[1], 1.0], dtype=np.float32)
    return spacing.astype(np.float32)


def get_inplane_spacing_hw(spacing):
    spacing = normalize_spacing(spacing)
    return np.array([float(spacing[0]), float(spacing[1])], dtype=np.float32)


def mask_to_surface(mask):
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return mask
    eroded = binary_erosion(mask)
    return mask ^ eroded


def compute_surface_distance_metrics(pred, gt, spacing_hw=None):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.sum() == 0 or gt.sum() == 0:
        return float("inf"), float("inf")

    pred_surface = mask_to_surface(pred)
    gt_surface = mask_to_surface(gt)
    if pred_surface.sum() == 0 or gt_surface.sum() == 0:
        return float("inf"), float("inf")

    sampling = None if spacing_hw is None else np.asarray(spacing_hw, dtype=np.float32)
    dt_gt = distance_transform_edt(~gt_surface, sampling=sampling)
    dt_pred = distance_transform_edt(~pred_surface, sampling=sampling)

    d_pred_to_gt = dt_gt[pred_surface]
    d_gt_to_pred = dt_pred[gt_surface]
    all_distances = np.concatenate([d_pred_to_gt, d_gt_to_pred], axis=0)
    if all_distances.size == 0:
        return float("inf"), float("inf")

    hd95 = float(np.percentile(all_distances, 95))
    asd = float(np.mean(all_distances))
    return hd95, asd


def evaluate_predictions(pred_dir, gt_dir):
    results = []
    pred_files = sorted(glob(join(pred_dir, "**", "*.npz"), recursive=True))
    print(f"Found {len(pred_files)} prediction files")

    for pred_file in tqdm(pred_files, desc="Evaluating"):
        pred_data = np.load(pred_file, allow_pickle=True)
        pred_segs = pred_data["segs"]
        spacing = (
            normalize_spacing(pred_data["spacing"])
            if "spacing" in pred_data
            else normalize_spacing([1.0, 1.0, 1.0])
        )

        file_name = basename(pred_file)
        gt_file = join(gt_dir, file_name)
        if not os.path.exists(gt_file):
            print(f"Warning: missing ground-truth file {gt_file}")
            continue

        gt_data = np.load(gt_file, allow_pickle=True)
        gt_segs = gt_data["gts"]

        if pred_segs.shape != gt_segs.shape:
            print(f"Warning: shape mismatch {pred_segs.shape} vs {gt_segs.shape} for {file_name}")
            continue

        slice_metrics = {
            "dice": [],
            "iou": [],
            "sensitivity": [],
            "specificity": [],
            "precision": [],
        }

        for slice_idx in range(pred_segs.shape[0]):
            pred_binary = (pred_segs[slice_idx] > 0).astype(np.float32)
            gt_binary = (gt_segs[slice_idx] > 0).astype(np.float32)
            is_empty_slice = np.sum(pred_binary) == 0 and np.sum(gt_binary) == 0
            if is_empty_slice:
                continue

            slice_metrics["dice"].append(compute_dice_score(pred_binary, gt_binary))
            slice_metrics["iou"].append(compute_iou(pred_binary, gt_binary))
            slice_metrics["sensitivity"].append(compute_sensitivity(pred_binary, gt_binary))
            slice_metrics["specificity"].append(compute_specificity(pred_binary, gt_binary))
            slice_metrics["precision"].append(compute_precision(pred_binary, gt_binary))

        pred_3d_binary = (pred_segs > 0).astype(np.float32)
        gt_3d_binary = (gt_segs > 0).astype(np.float32)

        dice_3d = compute_dice_score(pred_3d_binary, gt_3d_binary)
        iou_3d = compute_iou(pred_3d_binary, gt_3d_binary)

        spacing_3d_ordered = np.array(
            [float(spacing[2]), float(spacing[0]), float(spacing[1])], dtype=np.float32
        )
        hd95_3d_px, asd_3d_px = compute_surface_distance_metrics(
            pred_3d_binary.astype(bool), gt_3d_binary.astype(bool), spacing_hw=None
        )
        hd95_3d_mm, asd_3d_mm = compute_surface_distance_metrics(
            pred_3d_binary.astype(bool), gt_3d_binary.astype(bool), spacing_hw=spacing_3d_ordered
        )

        hd95_3d_px = float("nan") if hd95_3d_px == float("inf") else hd95_3d_px
        hd95_3d_mm = float("nan") if hd95_3d_mm == float("inf") else hd95_3d_mm
        asd_3d_px = float("nan") if asd_3d_px == float("inf") else asd_3d_px
        asd_3d_mm = float("nan") if asd_3d_mm == float("inf") else asd_3d_mm

        voxel_volume = float(np.prod(spacing[:3]))
        volume_pred = float(np.sum(pred_3d_binary) * voxel_volume)
        volume_gt = float(np.sum(gt_3d_binary) * voxel_volume)

        if volume_gt == 0:
            volume_error = 0.0 if volume_pred == 0 else float("nan")
            rvd = 0.0 if volume_pred == 0 else float("nan")
        else:
            volume_error = abs(volume_pred - volume_gt) / volume_gt * 100.0
            rvd = (volume_pred - volume_gt) / volume_gt * 100.0

        if slice_metrics["dice"]:
            dice_2d_mean = float(np.mean(slice_metrics["dice"]))
            dice_2d_std = float(np.std(slice_metrics["dice"]))
            iou_2d_mean = float(np.mean(slice_metrics["iou"]))
            iou_2d_std = float(np.std(slice_metrics["iou"]))
            sensitivity_2d_mean = float(np.mean(slice_metrics["sensitivity"]))
            sensitivity_2d_std = float(np.std(slice_metrics["sensitivity"]))
            specificity_2d_mean = float(np.mean(slice_metrics["specificity"]))
            specificity_2d_std = float(np.std(slice_metrics["specificity"]))
            precision_2d_mean = float(np.mean(slice_metrics["precision"]))
            precision_2d_std = float(np.std(slice_metrics["precision"]))
        else:
            dice_2d_mean = float("nan")
            dice_2d_std = float("nan")
            iou_2d_mean = float("nan")
            iou_2d_std = float("nan")
            sensitivity_2d_mean = float("nan")
            sensitivity_2d_std = float("nan")
            specificity_2d_mean = float("nan")
            specificity_2d_std = float("nan")
            precision_2d_mean = float("nan")
            precision_2d_std = float("nan")

        results.append(
            {
                "case": file_name,
                "Dice_3D_Overall": dice_3d,
                "IoU_3D_Overall": iou_3d,
                "Dice_2D_Mean": dice_2d_mean,
                "Dice_2D_Std": dice_2d_std,
                "IoU_2D_Mean": iou_2d_mean,
                "IoU_2D_Std": iou_2d_std,
                "Sensitivity_2D_Mean": sensitivity_2d_mean,
                "Sensitivity_2D_Std": sensitivity_2d_std,
                "Specificity_2D_Mean": specificity_2d_mean,
                "Specificity_2D_Std": specificity_2d_std,
                "Precision_2D_Mean": precision_2d_mean,
                "Precision_2D_Std": precision_2d_std,
                "HD95_3D_px": hd95_3d_px,
                "HD95_3D_mm": hd95_3d_mm,
                "ASD_3D_px": asd_3d_px,
                "ASD_3D_mm": asd_3d_mm,
                "Volume_Error_%": volume_error,
                "Relative_Volume_Diff_%": rvd,
                "Volume_Pred_mm3": volume_pred,
                "Volume_GT_mm3": volume_gt,
                "Num_Slices": pred_segs.shape[0],
                "Num_NonEmpty_Slices": len(slice_metrics["dice"]),
                "Spacing": f"{spacing[0]:.4f},{spacing[1]:.4f},{spacing[2]:.4f}",
            }
        )

    return results


def print_summary(df):
    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    print(f"Cases: {len(df)}")
    print(f"Total slices: {df['Num_Slices'].sum()}")
    print(f"Non-empty slices: {df['Num_NonEmpty_Slices'].sum()}")

    print("\n3D Metrics")
    print(f"Dice (3D): {df['Dice_3D_Overall'].mean():.4f} ± {df['Dice_3D_Overall'].std():.4f}")
    print(f"IoU  (3D): {df['IoU_3D_Overall'].mean():.4f} ± {df['IoU_3D_Overall'].std():.4f}")

    print("\n2D Metrics (averaged over non-empty slices)")
    print(f"Dice:       {df['Dice_2D_Mean'].mean():.4f} ± {df['Dice_2D_Mean'].std():.4f}")
    print(f"IoU:        {df['IoU_2D_Mean'].mean():.4f} ± {df['IoU_2D_Mean'].std():.4f}")
    print(f"Recall:     {df['Sensitivity_2D_Mean'].mean():.4f} ± {df['Sensitivity_2D_Mean'].std():.4f}")
    print(f"Specificity:{df['Specificity_2D_Mean'].mean():.4f} ± {df['Specificity_2D_Mean'].std():.4f}")
    print(f"Precision:  {df['Precision_2D_Mean'].mean():.4f} ± {df['Precision_2D_Mean'].std():.4f}")

    print("\nHD95 (3D volume)")
    print(f"HD95 px:    {df['HD95_3D_px'].mean():.4f} ± {df['HD95_3D_px'].std():.4f}")
    print(f"HD95 mm:    {df['HD95_3D_mm'].mean():.4f} ± {df['HD95_3D_mm'].std():.4f}")

    print("\nASD (3D volume)")
    print(f"ASD px:     {df['ASD_3D_px'].mean():.4f} ± {df['ASD_3D_px'].std():.4f}")
    print(f"ASD mm:     {df['ASD_3D_mm'].mean():.4f} ± {df['ASD_3D_mm'].std():.4f}")

    print("\nVolume")
    print(f"Volume Error: {df['Volume_Error_%'].mean():.4f} ± {df['Volume_Error_%'].std():.4f} %")
    print(
        f"RVD:          {df['Relative_Volume_Diff_%'].mean():.4f} ± "
        f"{df['Relative_Volume_Diff_%'].std():.4f} %"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate prediction `.npz` files against ground-truth `.npz` files."
    )
    parser.add_argument("-pred_dir", type=str, required=True, help="Directory containing prediction `.npz` files")
    parser.add_argument("-gt_dir", type=str, required=True, help="Directory containing ground-truth `.npz` files")
    parser.add_argument(
        "-output_csv",
        type=str,
        default="evaluation_results.csv",
        help="Path to the output CSV summary",
    )
    args = parser.parse_args()

    results = evaluate_predictions(args.pred_dir, args.gt_dir)
    if not results:
        print("No valid evaluation results were produced.")
        return

    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print_summary(df)
    print(f"\nSaved per-case metrics to: {args.output_csv}")


if __name__ == "__main__":
    main()
