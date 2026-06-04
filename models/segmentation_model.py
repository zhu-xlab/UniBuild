import torch
import torch.nn as nn

from .backbone.factory import build_backbone
from .decoder.factory import build_decoder


class SegmentationModel(nn.Module):
    def __init__(self, model_cfg, nclass):
        super().__init__()

        self.backbone = build_backbone(model_cfg["backbone"])

        decoder_in_channels = getattr(self.backbone, "feature_channels", None)
        if decoder_in_channels is None:
            decoder_in_channels = self.backbone.embed_dim

        self.decoder = build_decoder(
            model_cfg["decoder"],
            in_channels=decoder_in_channels,
            nclass=nclass,
            decoder_defaults=getattr(self.backbone, "decoder_defaults", None),
        )

        self.patch_size = getattr(self.backbone, "patch_size", 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        patch_h, patch_w = h // self.patch_size, w // self.patch_size

        feats = self.backbone.get_intermediate_layers(x)
        out = self.decoder(x, feats, image_hw=(h, w), patch_hw=(patch_h, patch_w))
        return out
