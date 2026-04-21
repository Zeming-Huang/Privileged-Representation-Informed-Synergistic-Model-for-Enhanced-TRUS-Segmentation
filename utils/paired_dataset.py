# -*- coding: utf-8 -*-
"""
Paired MRI/TRUS dataset utilities for PRISM training.

The dataset expects two directory roots with matching `imgs/` and `gts/`
subdirectories. Each TRUS slice is paired with the corresponding MRI slice
using a shared case identifier.
"""

import os
import random
from glob import glob
from os.path import basename, isfile, join, normpath

import numpy as np
import torch
from torch.utils.data import Dataset


class PairedNpyDataset(Dataset):
    """Load paired TRUS/MRI `.npy` slices for dual-modal training."""

    def __init__(self, trus_root, mri_root, image_size=256, bbox_shift=5, data_aug=False):
        if not os.path.isabs(trus_root):
            trus_root = os.path.abspath(trus_root)
        if not os.path.isabs(mri_root):
            mri_root = os.path.abspath(mri_root)

        self.trus_root = normpath(trus_root)
        self.mri_root = normpath(mri_root)
        self.image_size = image_size
        self.bbox_shift = bbox_shift
        self.data_aug = data_aug

        trus_gt_path = normpath(join(self.trus_root, "gts"))
        trus_img_path = normpath(join(self.trus_root, "imgs"))
        mri_gt_path = normpath(join(self.mri_root, "gts"))
        mri_img_path = normpath(join(self.mri_root, "imgs"))

        if not os.path.exists(trus_gt_path) or not os.path.exists(trus_img_path):
            raise ValueError(f"Missing TRUS dataset directories: {trus_gt_path}, {trus_img_path}")
        if not os.path.exists(mri_gt_path) or not os.path.exists(mri_img_path):
            raise ValueError(f"Missing MRI dataset directories: {mri_gt_path}, {mri_img_path}")

        self.trus_img_path = trus_img_path
        self.mri_gt_path = mri_gt_path
        self.mri_img_path = mri_img_path

        trus_gt_files = sorted(glob(join(trus_gt_path, "*.npy"), recursive=True))
        trus_gt_files = [
            file_path
            for file_path in trus_gt_files
            if isfile(normpath(join(self.trus_img_path, basename(file_path))))
        ]

        self.valid_pairs = []
        for trus_gt_file in trus_gt_files:
            trus_basename = basename(trus_gt_file)
            trus_img_file = normpath(join(self.trus_img_path, trus_basename))

            if trus_basename.startswith("TRUS_Prostate_"):
                mri_basename = trus_basename.replace("TRUS_Prostate_", "MRI_Prostate_")
            else:
                mri_basename = trus_basename

            mri_gt_file = normpath(join(self.mri_gt_path, mri_basename))
            mri_img_file = normpath(join(self.mri_img_path, mri_basename))

            if isfile(trus_img_file) and isfile(mri_gt_file) and isfile(mri_img_file):
                self.valid_pairs.append(
                    {
                        "trus_img": trus_img_file,
                        "trus_gt": trus_gt_file,
                        "mri_img": mri_img_file,
                        "mri_gt": mri_gt_file,
                    }
                )

        if not self.valid_pairs:
            raise ValueError("No valid paired MRI/TRUS samples were found.")

        print(f"Found {len(self.valid_pairs)} valid MRI/TRUS pairs")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, index):
        pair = self.valid_pairs[index]

        trus_img = np.load(pair["trus_img"], "r", allow_pickle=True)
        trus_gt = np.load(pair["trus_gt"], "r", allow_pickle=True)
        mri_img = np.load(pair["mri_img"], "r", allow_pickle=True)
        mri_gt = np.load(pair["mri_gt"], "r", allow_pickle=True)

        assert trus_img.shape[:2] == trus_gt.shape, (
            f"TRUS image/label shape mismatch: {trus_img.shape} vs {trus_gt.shape}"
        )
        assert mri_img.shape[:2] == mri_gt.shape, (
            f"MRI image/label shape mismatch: {mri_img.shape} vs {mri_gt.shape}"
        )

        trus_gt = np.uint8(trus_gt > 0)
        mri_gt = np.uint8(mri_gt > 0)

        if self.data_aug:
            if random.random() > 0.5:
                trus_img = np.ascontiguousarray(np.flip(trus_img, axis=1))
                trus_gt = np.ascontiguousarray(np.flip(trus_gt, axis=1))
                mri_img = np.ascontiguousarray(np.flip(mri_img, axis=1))
                mri_gt = np.ascontiguousarray(np.flip(mri_gt, axis=1))

            if random.random() > 0.5:
                trus_img = np.ascontiguousarray(np.flip(trus_img, axis=0))
                trus_gt = np.ascontiguousarray(np.flip(trus_gt, axis=0))
                mri_img = np.ascontiguousarray(np.flip(mri_img, axis=0))
                mri_gt = np.ascontiguousarray(np.flip(mri_gt, axis=0))

        trus_img_tensor = torch.tensor(trus_img).permute(2, 0, 1).float()
        mri_img_tensor = torch.tensor(mri_img).permute(2, 0, 1).float()

        assert torch.max(trus_img_tensor) <= 1.0 and torch.min(trus_img_tensor) >= 0.0, (
            f"TRUS image must be normalized to [0, 1], got "
            f"[{torch.min(trus_img_tensor):.3f}, {torch.max(trus_img_tensor):.3f}]"
        )
        assert torch.max(mri_img_tensor) <= 1.0 and torch.min(mri_img_tensor) >= 0.0, (
            f"MRI image must be normalized to [0, 1], got "
            f"[{torch.min(mri_img_tensor):.3f}, {torch.max(mri_img_tensor):.3f}]"
        )

        trus_gt_tensor = torch.tensor(trus_gt[None, :, :]).long()
        y_indices, x_indices = np.where(trus_gt > 0)

        if len(y_indices) > 0:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            height, width = trus_gt.shape
            x_min = max(0, x_min - random.randint(0, self.bbox_shift))
            x_max = min(width - 1, x_max + random.randint(0, self.bbox_shift))
            y_min = max(0, y_min - random.randint(0, self.bbox_shift))
            y_max = min(height - 1, y_max + random.randint(0, self.bbox_shift))
            bboxes = np.array([x_min, y_min, x_max, y_max])
        else:
            height, width = trus_gt.shape
            bboxes = np.array([0, 0, width - 1, height - 1])

        bboxes_tensor = torch.tensor(bboxes[None, None, ...]).float()

        return {
            "trus_image": trus_img_tensor,
            "mri_image": mri_img_tensor,
            "gt2D": trus_gt_tensor,
            "bboxes": bboxes_tensor,
            "image_name": basename(pair["trus_gt"]),
            "new_size": torch.tensor(np.array([self.image_size, self.image_size])).long(),
            "original_size": torch.tensor(np.array([self.image_size, self.image_size])).long(),
        }

    def get_sample_info(self, index):
        pair = self.valid_pairs[index]
        return {
            "trus_img": pair["trus_img"],
            "mri_img": pair["mri_img"],
            "trus_gt": pair["trus_gt"],
            "mri_gt": pair["mri_gt"],
        }
