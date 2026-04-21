#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Efficient paired dataset wrapper for PRISM training."""

import random

from torch.utils.data import Dataset

from .paired_dataset import PairedNpyDataset


class EfficientPairedNpyDataset(Dataset):
    """Sample a configurable subset of the paired dataset per epoch."""

    def __init__(
        self,
        trus_root,
        mri_root,
        image_size=256,
        bbox_shift=5,
        data_aug=False,
        samples_per_epoch=64,
    ):
        self.samples_per_epoch = float("inf") if samples_per_epoch is None else samples_per_epoch
        self.full_dataset = PairedNpyDataset(
            trus_root=trus_root,
            mri_root=mri_root,
            image_size=image_size,
            bbox_shift=bbox_shift,
            data_aug=data_aug,
        )
        self.total_samples = len(self.full_dataset)
        self.current_epoch_indices = self._generate_epoch_indices()

        actual_samples = min(len(self.current_epoch_indices), self.total_samples)
        utilization = actual_samples / self.total_samples * 100.0
        print("Initialized EfficientPairedNpyDataset")
        print(f"  - total samples: {self.total_samples}")
        print(f"  - samples per epoch: {actual_samples} ({utilization:.1f}%)")

    def _generate_epoch_indices(self):
        if self.samples_per_epoch == float("inf") or self.samples_per_epoch >= self.total_samples:
            indices = list(range(self.total_samples))
            random.shuffle(indices)
            return indices
        return random.sample(range(self.total_samples), self.samples_per_epoch)

    def __len__(self):
        return len(self.current_epoch_indices)

    def __getitem__(self, idx):
        return self.full_dataset[self.current_epoch_indices[idx]]

    def new_epoch(self):
        self.current_epoch_indices = self._generate_epoch_indices()
        print(f"Started a new epoch with {len(self.current_epoch_indices)} samples")

    def get_epoch_info(self):
        return {
            "total_samples": self.total_samples,
            "current_epoch_samples": len(self.current_epoch_indices),
            "utilization_rate": len(self.current_epoch_indices) / self.total_samples * 100.0,
            "indices": self.current_epoch_indices[:10],
        }
