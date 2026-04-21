#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the dual-modal PRISM model with paired TRUS and MRI inputs."""

import argparse


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train the dual-modal PRISM model with paired TRUS and MRI inputs."
    )
    parser.add_argument("-trus_data_root", type=str, default="./data/train/trus", help="Path to the TRUS training root.")
    parser.add_argument("-mri_data_root", type=str, default="./data/train/mri", help="Path to the MRI training root.")
    parser.add_argument("-val_trus_data_root", type=str, default="./data/val/trus", help="Path to the validation TRUS root.")
    parser.add_argument("-val_mri_data_root", type=str, default="./data/val/mri", help="Path to the validation MRI root.")
    parser.add_argument("-pretrained_checkpoint", type=str, default="./checkpoints/lite_medsam.pth", help="Path to the base LiteMedSAM checkpoint.")
    parser.add_argument("-trus_pretrained_checkpoint", type=str, default=None, help="Optional TRUS encoder checkpoint. Falls back to `-pretrained_checkpoint`.")
    parser.add_argument("-mri_pretrained_checkpoint", type=str, default=None, help="Optional MRI encoder checkpoint. Falls back to `-pretrained_checkpoint`.")
    parser.add_argument("-resume", type=str, default=None, help="Optional checkpoint used to resume training.")
    parser.add_argument("-work_dir", type=str, default="./work_dir/prism", help="Directory where checkpoints and logs will be saved.")
    parser.add_argument("-num_epochs", type=int, default=20, help="Number of epochs to train.")
    parser.add_argument("-batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("-num_workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("-device", type=str, default="cuda:0", help="Device to train on.")
    parser.add_argument("-bbox_shift", type=int, default=5, help="Bounding-box perturbation used during training.")
    parser.add_argument("-lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("-weight_decay", type=float, default=1e-2, help="Weight decay.")
    parser.add_argument("-iou_loss_weight", type=float, default=1.0, help="Weight of the IoU regression loss.")
    parser.add_argument("-seg_loss_weight", type=float, default=1.0, help="Weight of the Dice segmentation loss.")
    parser.add_argument("-ce_loss_weight", type=float, default=1.0, help="Weight of the BCE segmentation loss.")
    parser.add_argument("-mmd_loss_weight", type=float, default=0.1, help="Weight of the cross-modal MMD alignment loss.")
    parser.add_argument("-val_interval", type=int, default=1, help="Run validation every N epochs.")
    parser.add_argument("-early_stopping_patience", type=int, default=20, help="Early stopping patience.")
    parser.add_argument("-min_delta", type=float, default=1e-4, help="Minimum Dice improvement for early stopping.")
    parser.add_argument("-lr_scheduler", type=str, default="plateau", choices=["plateau", "cosine", "step"], help="Learning-rate scheduler type.")
    parser.add_argument("-samples_per_epoch", type=int, default=64, help="Number of paired samples drawn per epoch by the efficient dataset wrapper.")
    parser.add_argument("-dropout", type=float, default=0.0, help="Reserved dropout argument for compatibility.")
    parser.add_argument("--use_adaptive_fusion", action="store_true", default=True, help="Use adaptive fusion.")
    parser.add_argument("--no_adaptive_fusion", action="store_false", dest="use_adaptive_fusion", help="Disable adaptive fusion and use linear fusion instead.")
    parser.add_argument("--no_fusion", action="store_true", default=False, help="Disable fusion and use enhanced features directly.")
    parser.add_argument("--ablation_no_mmd", action="store_true", default=False, help="Disable the MMD alignment loss.")
    parser.add_argument("-freeze_encoders", action="store_true", help="Freeze the TRUS and MRI encoders during training.")
    parser.add_argument("-mixed_precision", action="store_true", help="Enable mixed precision training.")
    parser.add_argument("--sanity_check", action="store_true", help="Run the data-loading sanity check before training.")
    return parser


def main():
    args = build_parser().parse_args()

    import os
    from pathlib import Path

    import monai
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    from enhanced_dual_modal import EnhancedDualModalMedSAM_Lite, EnhancedMaskDecoder
    from segment_anything.modeling import PromptEncoder, TwoWayTransformer
    from tiny_vit_sam import TinyViT
    from utils.efficient_paired_dataset import EfficientPairedNpyDataset
    from utils.paired_dataset import PairedNpyDataset

    def compute_iou_target(pred_logits, gt_mask):
        pred_binary = (torch.sigmoid(pred_logits) > 0.5)
        gt_binary = gt_mask.bool()
        intersection = torch.count_nonzero(pred_binary & gt_binary, dim=(1, 2, 3))
        union = torch.count_nonzero(pred_binary | gt_binary, dim=(1, 2, 3))
        iou = torch.where(union == 0, torch.ones_like(union, dtype=torch.float32), intersection.float() / union.float())
        return iou.unsqueeze(1)

    @torch.no_grad()
    def compute_dice_score(pred_logits, gt_mask):
        pred = (torch.sigmoid(pred_logits) > 0.5).float()
        gt_mask = gt_mask.float()
        intersection = torch.sum(pred * gt_mask, dim=(1, 2, 3))
        union = torch.sum(pred, dim=(1, 2, 3)) + torch.sum(gt_mask, dim=(1, 2, 3))
        dice = torch.where(
            union == 0,
            torch.where(intersection == 0, torch.ones_like(intersection), torch.zeros_like(intersection)),
            2.0 * intersection / union,
        )
        return float(dice.mean().item())

    @torch.no_grad()
    def evaluate_on_val(model, val_trus_root, val_mri_root, batch_size, num_workers, device):
        if not val_trus_root or not val_mri_root:
            return float("nan")
        if not os.path.isdir(val_trus_root) or not os.path.isdir(val_mri_root):
            return float("nan")

        model.eval()
        dataset = PairedNpyDataset(val_trus_root, val_mri_root, data_aug=False)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")

        dices = []
        for batch in loader:
            trus_image = batch["trus_image"].to(device)
            mri_image = batch["mri_image"].to(device)
            gt2d = batch["gt2D"].to(device).float()
            boxes = batch["bboxes"].to(device)
            logits_pred, _ = model(trus_image, mri_image, boxes, training=False)
            dices.append(compute_dice_score(logits_pred, gt2d))

        model.train()
        if not dices:
            return float("nan")
        return float(np.mean(dices))

    def build_model(mmd_loss_weight):
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
            transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_src_enhancement=True,
        )
        model = EnhancedDualModalMedSAM_Lite(
            image_encoder=image_encoder,
            mask_decoder=mask_decoder,
            prompt_encoder=prompt_encoder,
            use_cross_modal=True,
            use_src_enhancement=True,
            use_adaptive_fusion=args.use_adaptive_fusion,
            use_fusion=not args.no_fusion,
        )
        if hasattr(model, "cross_modal_extractor"):
            model.cross_modal_extractor.mmd_weight = mmd_loss_weight
        return model

    def load_pretrained_weights(model, checkpoint_path, component_name):
        if not checkpoint_path or not os.path.isfile(checkpoint_path):
            print(f"Skipping missing checkpoint for {component_name}: {checkpoint_path}")
            return

        print(f"Loading {component_name} pretrained weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

        if component_name == "TRUS":
            model.load_state_dict(
                {
                    key.replace("image_encoder.", "trus_encoder."): value
                    for key, value in state_dict.items()
                    if key.startswith("image_encoder.")
                },
                strict=False,
            )
        elif component_name == "MRI":
            model.load_state_dict(
                {
                    key.replace("image_encoder.", "mri_encoder."): value
                    for key, value in state_dict.items()
                    if key.startswith("image_encoder.")
                },
                strict=False,
            )
        elif component_name == "SAM":
            model.load_state_dict(
                {
                    key: value
                    for key, value in state_dict.items()
                    if key.startswith(("mask_decoder.", "prompt_encoder."))
                },
                strict=False,
            )

    def configure_optimizer(model):
        if args.freeze_encoders:
            for param in model.trus_encoder.parameters():
                param.requires_grad = False
            if model.mri_encoder is not None:
                for param in model.mri_encoder.parameters():
                    param.requires_grad = False

            trainable_params = list(model.cross_modal_extractor.parameters())
            trainable_params += list(model.mask_decoder.parameters())
            trainable_params += list(model.prompt_encoder.parameters())
            optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        else:
            mri_params = []
            if model.mri_encoder is not None and model.mri_encoder is not model.trus_encoder:
                mri_params = list(model.mri_encoder.parameters())

            optimizer = optim.AdamW(
                [
                    {"params": model.cross_modal_extractor.parameters(), "lr": args.lr},
                    {"params": model.trus_encoder.parameters(), "lr": args.lr * 0.1},
                    {"params": mri_params, "lr": args.lr * 0.1},
                    {"params": model.mask_decoder.parameters(), "lr": args.lr * 0.1},
                    {"params": model.prompt_encoder.parameters(), "lr": args.lr * 0.1},
                ],
                weight_decay=args.weight_decay,
            )
        return optimizer

    def configure_scheduler(optimizer):
        if args.lr_scheduler == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.7, patience=5, cooldown=2, min_lr=1e-7)
        if args.lr_scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-7)
        return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or "cpu" not in args.device else "cpu")

    if args.ablation_no_mmd:
        print("[ABLATION] Disabled the MMD loss by forcing its weight to 0.")
    if args.no_fusion:
        print("[ABLATION] Disabled fusion and using enhanced features directly.")

    effective_mmd_weight = 0.0 if args.ablation_no_mmd else args.mmd_loss_weight
    trus_checkpoint = args.trus_pretrained_checkpoint or args.pretrained_checkpoint
    mri_checkpoint = args.mri_pretrained_checkpoint or args.pretrained_checkpoint

    model = build_model(effective_mmd_weight).to(device)
    load_pretrained_weights(model, trus_checkpoint, "TRUS")
    load_pretrained_weights(model, mri_checkpoint, "MRI")
    load_pretrained_weights(model, args.pretrained_checkpoint, "SAM")

    if args.sanity_check:
        sanity_dataset = PairedNpyDataset(args.trus_data_root, args.mri_data_root, bbox_shift=args.bbox_shift, data_aug=False)
        sanity_sample = sanity_dataset[0]
        print("Sanity check passed:")
        print(f"  trus_image: {tuple(sanity_sample['trus_image'].shape)}")
        print(f"  mri_image: {tuple(sanity_sample['mri_image'].shape)}")
        print(f"  gt2D: {tuple(sanity_sample['gt2D'].shape)}")
        print(f"  bboxes: {tuple(sanity_sample['bboxes'].shape)}")
        return

    train_dataset = EfficientPairedNpyDataset(
        args.trus_data_root,
        args.mri_data_root,
        bbox_shift=args.bbox_shift,
        data_aug=False,
        samples_per_epoch=args.samples_per_epoch,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    optimizer = configure_optimizer(model)
    scheduler = configure_scheduler(optimizer)
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    iou_loss = nn.MSELoss(reduction="mean")

    scaler = None
    autocast_context = None
    if args.mixed_precision:
        from torch.amp import GradScaler, autocast

        scaler = GradScaler(device.type if device.type in {"cuda", "cpu"} else "cuda")
        autocast_context = autocast

    start_epoch = 0
    best_val_dice = float("-inf")
    epochs_without_improvement = 0

    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from checkpoint {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_dice = checkpoint.get("best_val_dice", float("-inf"))
        print(f"Loaded checkpoint from epoch {start_epoch}")

    print(f"Model parameters: {sum(param.numel() for param in model.parameters())}")

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        if hasattr(train_dataset, "new_epoch"):
            train_dataset.new_epoch()

        epoch_losses = []
        epoch_dices = []

        for batch in train_loader:
            trus_image = batch["trus_image"].to(device)
            mri_image = batch["mri_image"].to(device)
            gt2d = batch["gt2D"].to(device).float()
            boxes = batch["bboxes"].to(device)

            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with autocast_context(device_type=device.type):
                    logits_pred, iou_pred, mmd_loss = model(trus_image, mri_image, boxes, training=True)
                    loss_seg = seg_loss(logits_pred, gt2d)
                    loss_ce = ce_loss(logits_pred, gt2d)
                    loss_iou = iou_loss(iou_pred, compute_iou_target(logits_pred, gt2d))
                    total_loss = (
                        args.seg_loss_weight * loss_seg
                        + args.ce_loss_weight * loss_ce
                        + args.iou_loss_weight * loss_iou
                        + effective_mmd_weight * mmd_loss
                    )
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits_pred, iou_pred, mmd_loss = model(trus_image, mri_image, boxes, training=True)
                loss_seg = seg_loss(logits_pred, gt2d)
                loss_ce = ce_loss(logits_pred, gt2d)
                loss_iou = iou_loss(iou_pred, compute_iou_target(logits_pred, gt2d))
                total_loss = (
                    args.seg_loss_weight * loss_seg
                    + args.ce_loss_weight * loss_ce
                    + args.iou_loss_weight * loss_iou
                    + effective_mmd_weight * mmd_loss
                )
                total_loss.backward()
                optimizer.step()

            epoch_losses.append(float(total_loss.item()))
            epoch_dices.append(compute_dice_score(logits_pred.detach(), gt2d))

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        train_dice = float(np.mean(epoch_dices)) if epoch_dices else float("nan")

        val_dice = float("nan")
        if args.val_interval > 0 and ((epoch + 1) % args.val_interval == 0):
            val_dice = evaluate_on_val(
                model=model,
                val_trus_root=args.val_trus_data_root,
                val_mri_root=args.val_mri_data_root,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
            )

        if args.lr_scheduler == "plateau":
            scheduler.step(val_dice if not np.isnan(val_dice) else train_dice)
        else:
            scheduler.step()

        print(
            f"Epoch {epoch + 1:03d}/{args.num_epochs:03d} | "
            f"train_loss={train_loss:.4f} | train_dice={train_dice:.4f} | val_dice={val_dice:.4f}"
        )

        latest_checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_dice": best_val_dice,
            "args": vars(args),
        }
        torch.save(latest_checkpoint, work_dir / "dual_modal_latest.pth")

        improved = not np.isnan(val_dice) and val_dice > (best_val_dice + args.min_delta)
        if improved:
            best_val_dice = val_dice
            epochs_without_improvement = 0
            best_checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_dice": best_val_dice,
                "args": vars(args),
            }
            torch.save(best_checkpoint, work_dir / "dual_modal_best.pth")
        elif not np.isnan(val_dice):
            epochs_without_improvement += 1

        if args.val_interval > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print("Early stopping triggered.")
            break


if __name__ == "__main__":
    main()
