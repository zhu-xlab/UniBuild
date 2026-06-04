import torch
import torch.nn as nn
import torch.nn.functional as F

from ..util.blocks import FeatureFusionBlock, _make_scratch


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=False,
        size=size,
    )


class HLRDPTDecoder(nn.Module):
    def __init__(
        self,
        nclass,
        in_channels,
        features=128,
        use_bn=False,
        out_channels=(96, 192, 384, 768),
        shallow_in_channels=3,
        shallow_channels=64,
    ):
        super().__init__()

        assert len(out_channels) == 4

        # --------------------------------------------------
        # 1) high-resolution shallow feature (1/2)
        # --------------------------------------------------
        self.shallow_extractor = nn.Sequential(
            nn.Conv2d(shallow_in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),  # 1/2
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, shallow_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(shallow_channels),
            nn.ReLU(inplace=True),
        )

        # --------------------------------------------------
        # 2) transformer feature projection + resize
        # layer_1 -> 1/4
        # layer_2 -> 1/8
        # layer_3 -> 1/16
        # layer_4 -> 1/32
        # --------------------------------------------------
        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels, oc, kernel_size=1, stride=1, padding=0)
            for oc in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),  # -> 1/4
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),  # -> 1/8
            nn.Identity(),                                                                              # -> 1/16
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),          # -> 1/32
        ])

        # --------------------------------------------------
        # 3) semantic-guided refinement on high-res shallow feature
        # all semantic features are projected to shallow_channels
        # and resized to shallow_hr resolution (1/2)
        # --------------------------------------------------
        self.semantic_reduces = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(oc, shallow_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(shallow_channels),
                nn.ReLU(inplace=True),
            )
            for oc in out_channels
        ])

        self.guide_fusions = nn.ModuleList([
            self._make_guide_fusion(shallow_channels)
            for _ in range(4)
        ])

        # --------------------------------------------------
        # 4) build guided shallow pyramid from semantically guided shallow_hr
        # guided_shallow_1: 1/2
        # guided_shallow_2: 1/4
        # guided_shallow_3: 1/8
        # guided_shallow_4: 1/16
        # --------------------------------------------------
        self.guided_down2 = nn.Sequential(
            nn.Conv2d(shallow_channels, shallow_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(shallow_channels),
            nn.ReLU(inplace=True),
        )

        self.guided_down3 = nn.Sequential(
            nn.Conv2d(shallow_channels, shallow_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(shallow_channels),
            nn.ReLU(inplace=True),
        )

        self.guided_down4 = nn.Sequential(
            nn.Conv2d(shallow_channels, shallow_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(shallow_channels),
            nn.ReLU(inplace=True),
        )

        # --------------------------------------------------
        # 5) standard DPT decoder backbone
        # --------------------------------------------------
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        # --------------------------------------------------
        # 6) guided fusion for decoder
        # path_4: 1/16 <- guided_shallow_4
        # path_3: 1/8  <- guided_shallow_3
        # path_2: 1/4  <- guided_shallow_2
        # path_1: 1/2  <- guided_shallow_1
        # --------------------------------------------------
        self.guided_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(shallow_channels, features, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True),
            )
            for _ in range(4)
        ])

        self.guided_gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(features + shallow_channels, features, kernel_size=1, stride=1, padding=0, bias=True),
                nn.Sigmoid(),
            )
            for _ in range(4)
        ])

        # --------------------------------------------------
        # 7) output head
        # --------------------------------------------------
        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features, nclass, kernel_size=1, stride=1, padding=0),
        )

    def _make_guide_fusion(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def _apply_guided_fusion(self, path, shallow_feat, proj_layer, gate_layer):
        if shallow_feat.shape[2:] != path.shape[2:]:
            shallow_feat = F.interpolate(
                shallow_feat,
                size=path.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        guided_feat = proj_layer(shallow_feat)
        gate = gate_layer(torch.cat([path, shallow_feat], dim=1))
        return path + (1.0 + gate) * guided_feat

    def forward(self, img, feats, image_hw, patch_hw):
        """
        img:   [B, 3, H, W]
        feats: list[Tensor], length=4
              each item shape:
              - [B, N, C]
              - or [B, C, H, W]
        image_hw: (H, W)
        patch_hw: (Hp, Wp)
        """
        patch_h, patch_w = patch_hw

        # --------------------------------------------------
        # 1) high-resolution shallow feature
        # --------------------------------------------------
        shallow_hr = self.shallow_extractor(img)   # 1/2

        # --------------------------------------------------
        # 2) transformer multi-scale features
        # --------------------------------------------------
        outs = []
        for i, x in enumerate(feats):
            if x.dim() == 3:
                x = x.permute(0, 2, 1).reshape(x.shape[0], x.shape[-1], patch_h, patch_w)

            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            outs.append(x)

        layer_1, layer_2, layer_3, layer_4 = outs
        # layer_1: 1/4
        # layer_2: 1/8
        # layer_3: 1/16
        # layer_4: 1/32

        # --------------------------------------------------
        # 3) semantic-guided high-resolution shallow feature
        # all semantic features are resized to shallow_hr (1/2)
        # --------------------------------------------------
        guided_shallow_hr = shallow_hr
        for feat, reduce_module, fusion_module in zip(
            outs, self.semantic_reduces, self.guide_fusions
        ):
            semantic = reduce_module(feat)
            semantic = F.interpolate(
                semantic,
                size=guided_shallow_hr.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            fused = fusion_module(torch.cat([guided_shallow_hr, semantic], dim=1))
            guided_shallow_hr = guided_shallow_hr + fused

        # --------------------------------------------------
        # 4) build guided shallow pyramid from guided_shallow_hr
        # --------------------------------------------------
        guided_shallow_1 = guided_shallow_hr                     # 1/2
        guided_shallow_2 = self.guided_down2(guided_shallow_1)  # 1/4
        guided_shallow_3 = self.guided_down3(guided_shallow_2)  # 1/8
        guided_shallow_4 = self.guided_down4(guided_shallow_3)  # 1/16

        # --------------------------------------------------
        # 5) DPT decode
        # --------------------------------------------------
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        # path_4: 1/32 -> 1/16, guided by guided_shallow_4
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_4 = self._apply_guided_fusion(
            path_4,
            guided_shallow_4,
            self.guided_projs[0],
            self.guided_gates[0],
        )

        # path_3: 1/16 -> 1/8, guided by guided_shallow_3
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_3 = self._apply_guided_fusion(
            path_3,
            guided_shallow_3,
            self.guided_projs[1],
            self.guided_gates[1],
        )

        # path_2: 1/8 -> 1/4, guided by guided_shallow_2
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_2 = self._apply_guided_fusion(
            path_2,
            guided_shallow_2,
            self.guided_projs[2],
            self.guided_gates[2],
        )

        # path_1: 1/4 -> 1/2, guided by guided_shallow_1
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        path_1 = self._apply_guided_fusion(
            path_1,
            guided_shallow_1,
            self.guided_projs[3],
            self.guided_gates[3],
        )

        # --------------------------------------------------
        # 6) output
        # --------------------------------------------------
        out = self.scratch.output_conv(path_1)
        out = F.interpolate(out, size=image_hw, mode="bilinear", align_corners=False)
        return out
    

