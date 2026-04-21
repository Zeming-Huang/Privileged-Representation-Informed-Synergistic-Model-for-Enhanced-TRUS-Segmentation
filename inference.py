#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public PRISM inference script for slice-level same-distribution inputs."""

import argparse
import os
import re
from collections import defaultdict
from glob import glob
from os.path import basename, isdir, isfile, join

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from enhanced_dual_modal import EnhancedDualModalMedSAM_Lite, EnhancedMaskDecoder
from prism_checkpoint_utils import remap_legacy_prism_state_dict_keys
from segment_anything.modeling import PromptEncoder, TwoWayTransformer
from tiny_vit_sam import TinyViT


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run PRISM inference on slice-level `imgs/*.npy` inputs. "
            "This public entry point uses the self-attention inference path only."
        )
    )
    parser.add_argument(
        "-data_root",
        type=str,
        required=True,
        help="Directory containing `imgs/`, `gts/`, and optional case-level `.npz` files.",
    )
    parser.add_argument(
        "-checkpoint",
        type=str,
        required=True,
        help="Path to the PRISM checkpoint.",
    )
    parser.add_argument(
        "-pred_save_dir",
        type=str,
        required=True,
        help="Directory where case-level `.npz` predictions will be saved.",
    )
    parser.add_argument(
        "-device",
        type=str,
        default="cuda:0",
        help="Device for inference, for example `cuda:0` or `cpu`.",
    )
    parser.add_argument(
        "-bbox_shift",
        type=int,
        default=5,
        help="Bounding-box expansion applied around the ground-truth foreground.",
    )
    return parser


def get_bbox(mask, bbox_shift=5):
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0:
        return np.array([0, 0, mask.shape[1] - 1, mask.shape[0] - 1], dtype=np.float32)

    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    height, width = mask.shape
    x_min = max(0, x_min - bbox_shift)
    x_max = min(width - 1, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(height - 1, y_max + bbox_shift)
    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)


def build_model():
    image_encoder = TinyViT(
        img_size=256,
        in_chans=3,
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_checkpoint=False,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        layer_lr_decay=0.8,
    )
    prompt_encoder = PromptEncoder(
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(256, 256),
        mask_in_chans=16,
    )
    mask_decoder = EnhancedMaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=256,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=256,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        use_src_enhancement=True,
    )
    return EnhancedDualModalMedSAM_Lite(
        image_encoder=image_encoder,
        mask_decoder=mask_decoder,
        prompt_encoder=prompt_encoder,
        use_cross_modal=True,
        use_src_enhancement=True,
    )


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    state_dict = remap_legacy_prism_state_dict_keys(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    return checkpoint


def group_case_slices(img_dir, gt_dir):
    case_slices = defaultdict(list)
    for img_path in sorted(glob(join(img_dir, "*.npy"))):
        file_name = basename(img_path)
        match = re.match(r"(TRUS_Prostate_case\d+)-(\d+)\.npy", file_name)
        if not match:
            continue
        case_name, slice_idx = match.group(1), int(match.group(2))
        gt_path = join(gt_dir, file_name)
        if not isfile(gt_path):
            continue
        case_slices[case_name].append((slice_idx, img_path, gt_path))

    for case_name in case_slices:
        case_slices[case_name] = sorted(case_slices[case_name], key=lambda item: item[0])
    return dict(case_slices)


@torch.no_grad()
def infer_case(model, slices, case_gts, device, bbox_shift):
    preds = []
    gts = []
    for slice_position, (_, img_path, gt_path) in enumerate(slices):
        img = np.load(img_path, allow_pickle=True)
        gt = np.load(gt_path, allow_pickle=True)
        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=-1)

        img_tensor = torch.tensor(img).float().permute(2, 0, 1).unsqueeze(0).to(device)
        gt_binary = (gt > 0).astype(np.uint8)
        box = torch.tensor(get_bbox(gt_binary, bbox_shift)[None, None, :], dtype=torch.float32, device=device)

        trus_feat = model.trus_encoder(img_tensor)
        image_embedding, _ = model.cross_modal_extractor(trus_feat, trus_feat)
        sparse_embeddings, dense_embeddings = model.prompt_encoder(points=None, boxes=box, masks=None)
        low_res_masks, _ = model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        pred = torch.sigmoid(low_res_masks).squeeze().cpu().numpy()
        pred = (pred > 0.5).astype(np.uint8)

        original_gt = case_gts[slice_position].astype(np.uint8)
        original_h, original_w = original_gt.shape
        if pred.shape != (original_h, original_w):
            pred = cv2.resize(pred, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

        preds.append(pred)
        gts.append(original_gt)

    return np.stack(preds, axis=0), np.stack(gts, axis=0)


def resolve_spacing(data_root, case_name, num_slices):
    case_npz = join(data_root, f"{case_name}.npz")
    if isfile(case_npz):
        case_data = np.load(case_npz, allow_pickle=True)
        if "spacing" in case_data:
            return case_data["spacing"]
    return np.array([1.0, 1.0, 1.0], dtype=np.float32)


def main():
    args = build_parser().parse_args()
    img_dir = join(args.data_root, "imgs")
    gt_dir = join(args.data_root, "gts")
    if not isdir(img_dir) or not isdir(gt_dir):
        raise FileNotFoundError(f"Expected `imgs/` and `gts/` inside {args.data_root}")

    os.makedirs(args.pred_save_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    model = build_model().to(device)
    load_checkpoint(model, args.checkpoint)
    model.eval()

    case_slices = group_case_slices(img_dir, gt_dir)
    print(f"Found {len(case_slices)} cases")

    for case_name, slices in tqdm(case_slices.items(), desc="Running inference"):
        case_npz_path = join(args.data_root, f"{case_name}.npz")
        case_npz = np.load(case_npz_path, allow_pickle=True) if isfile(case_npz_path) else None
        case_gts = case_npz["gts"] if case_npz is not None and "gts" in case_npz else None
        if case_gts is None:
            raise FileNotFoundError(f"Expected case-level npz with `gts` for {case_name} at {case_npz_path}")

        preds, gts = infer_case(model, slices, case_gts, device, args.bbox_shift)
        spacing = resolve_spacing(args.data_root, case_name, preds.shape[0])
        output_path = join(args.pred_save_dir, f"{case_name}.npz")
        np.savez_compressed(output_path, segs=preds, gts=gts, spacing=spacing)

    print(f"Saved predictions to: {args.pred_save_dir}")


if __name__ == "__main__":
    main()
