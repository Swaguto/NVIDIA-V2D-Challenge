# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the full frozen-trajectory Dex3 capsule gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "source/robotic_grounding"))
from robotic_grounding.retarget.track3_penetration import (  # noqa: E402
    check_penetrations,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--retarget", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-unlocked-legacy-model", action="store_true")
    parser.add_argument(
        "--assets",
        type=Path,
        default=REPO / "source/robotic_grounding/robotic_grounding/assets",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            check_penetrations(
                args.bundle,
                args.retarget,
                args.assets,
                args.out,
                allow_unlocked_legacy_model=args.allow_unlocked_legacy_model,
            ),
            indent=2,
        )
    )
