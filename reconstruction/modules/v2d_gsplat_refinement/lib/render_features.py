"""Deterministic feature padding for gsplat's supported CUDA channel counts."""
import torch
import torch.nn.functional as F

SUPPORTED_CHANNELS = (1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512)


def pad_render_features(features: torch.Tensor) -> torch.Tensor:
    """Avoid gsplat 1.4's uninitialized padding for e.g. RGB + four classes.

    Even unused channels participate in the CUDA alpha/distortion backward
    arithmetic: a zero output gradient cannot remove NaNs in those channels.
    Passing a supported count with explicit zeros avoids allocator-dependent
    gradient failures without discarding or changing any supervised feature.
    """
    channels = features.shape[-1]
    if channels < 1 or channels > SUPPORTED_CHANNELS[-1]:
        raise ValueError(f"Unsupported number of render features: {channels}")
    padded = next(count for count in SUPPORTED_CHANNELS if count >= channels)
    return F.pad(features, (0, padded - channels), value=0.0) if padded != channels else features
