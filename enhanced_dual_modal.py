import torch
import torch.nn as nn
from copy import deepcopy

from segment_anything.modeling import MaskDecoder

from tiny_vit_sam import CrossModalFeatureExtractor


class EnhancedMaskDecoder(MaskDecoder):
    """Thin wrapper that keeps the standard SAM decoder interface intact."""

    def __init__(self, use_src_enhancement=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_src_enhancement = use_src_enhancement

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ):
        return super().forward(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=multimask_output,
        )


class EnhancedDualModalMedSAM_Lite(nn.Module):
    """
    Dual-modal PRISM model.

    During training, MRI features guide TRUS feature learning through the
    cross-modal extractor. During inference, the model runs with TRUS inputs
    only and reuses the learned cross-modal module in self-guided mode.
    """

    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
        use_cross_modal=True,
        use_src_enhancement=True,
        use_adaptive_fusion=True,
        use_fusion=True,
    ):
        super().__init__()
        self.use_cross_modal = use_cross_modal
        self.use_src_enhancement = use_src_enhancement
        self.use_adaptive_fusion = use_adaptive_fusion
        self.use_fusion = use_fusion

        self.trus_encoder = image_encoder
        self.mri_encoder = deepcopy(image_encoder) if use_cross_modal else None

        if use_cross_modal:
            self.cross_modal_extractor = CrossModalFeatureExtractor(
                in_channels=256,
                num_heads=8,
                mmd_weight=0.1,
                use_adaptive_fusion=use_adaptive_fusion,
                use_fusion=use_fusion,
            )

        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

    def forward(self, trus_image, mri_image=None, boxes=None, training=False):
        trus_feat = self.trus_encoder(trus_image)

        if training and self.use_cross_modal and mri_image is not None:
            mri_feat = self.mri_encoder(mri_image)
            image_embedding, mmd_loss = self.cross_modal_extractor(trus_feat, mri_feat)
        elif self.use_cross_modal:
            image_embedding, _ = self.cross_modal_extractor(trus_feat, trus_feat)
            mmd_loss = None
        else:
            image_embedding = trus_feat
            mmd_loss = None

        if boxes is not None:
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=boxes,
                masks=None,
            )
        else:
            sparse_embeddings, dense_embeddings = None, None

        if sparse_embeddings is not None and dense_embeddings is not None:
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
        else:
            low_res_masks, iou_predictions = None, None

        if training and mmd_loss is not None:
            return low_res_masks, iou_predictions, mmd_loss
        return low_res_masks, iou_predictions

    @torch.no_grad()
    def postprocess_masks(self, masks, new_size, original_size):
        masks = masks[:, :, : new_size[0], : new_size[1]]
        masks = torch.nn.functional.interpolate(
            masks,
            size=(original_size[0], original_size[1]),
            mode="bilinear",
            align_corners=False,
        )
        return masks
