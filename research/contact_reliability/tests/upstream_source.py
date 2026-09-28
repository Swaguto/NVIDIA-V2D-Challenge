"""Locate NVIDIA source in this monorepo or the standalone research checkout."""
from pathlib import Path


def upstream_root():
    package = Path(__file__).resolve().parents[1]
    candidates = [package.parents[1], package / "upstream/video_to_data"]
    for root in candidates:
        if (root / "robotic_grounding/source/robotic_grounding").is_dir():
            return root
    return candidates[-1]
