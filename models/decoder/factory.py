from .dpt_decoder import HLRDPTDecoder


def build_decoder(cfg, in_channels, nclass, decoder_defaults=None):
    name = cfg["name"].lower()
    kwargs = cfg.get("kwargs", {})

    if decoder_defaults is not None:
        merged_kwargs = dict(decoder_defaults)
        merged_kwargs.update(kwargs)
        kwargs = merged_kwargs

    if name != "hlrdpt":
        raise ValueError(f"This standalone inference package only supports HLRDPT. Got: {name}")

    kwargs.setdefault("shallow_in_channels", 3)
    kwargs.setdefault("shallow_channels", 64)
    return HLRDPTDecoder(nclass=nclass, in_channels=in_channels, **kwargs)
