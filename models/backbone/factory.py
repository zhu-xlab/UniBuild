from .dinov3_backbone import DinoV3Backbone


def build_backbone(cfg):
    name = cfg["name"].lower()
    variant = cfg["variant"]
    out_indices = cfg.get("out_indices", None)

    if name == "dinov3":
        backbone = DinoV3Backbone(
            variant=variant,
            out_indices=out_indices,
            pretrained=cfg.get("pretrained", None),
        )
    else:
        raise ValueError(f"This standalone inference package only supports DINOv3. Got: {name}")

    return backbone
