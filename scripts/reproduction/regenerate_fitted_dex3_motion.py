"""Regenerate selected motion from verified fitted inputs; no upstream refit."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from reproduce_frozen_dex3_replay import digest, restore_methods


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fitted-archive", type=Path, required=True)
    parser.add_argument("--fitted-manifest", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Use a fresh output directory")
    case = args.case.resolve()
    case_manifest = json.loads((case / "file-manifest.json").read_text())
    for name, expected in case_manifest["files_sha256"].items():
        if digest(case / name) != expected:
            raise ValueError("Frozen case hash mismatch: " + name)
    args.out.mkdir(parents=True)
    out = args.out.resolve()
    snapshot = (
        Path(__file__).resolve().parents[2]
        / "docs/team/reproduction/dex3-motion-replay"
    )
    restore_methods(
        snapshot / "methods.tar.gz", snapshot / "methods-manifest.json", out / "methods"
    )
    restore_methods(args.fitted_archive, args.fitted_manifest, out / "inputs")
    inputs = out / "inputs"
    target = inputs / "regenerated-selected-motion"
    subprocess.run(
        [
            sys.executable,
            str(out / "methods/clearance_gated_edge_motion_v2.py"),
            "--source",
            str(case / "source/robotic_grounding"),
            "--bundle",
            str(case / "bundle"),
            "--baseline",
            str(case / "retarget"),
            "--parent-proposal",
            str(inputs / "global-free-segment-motion-v3"),
            "--interval-inputs",
            str(inputs / "contact-interval-dense-full-chain-inputs-v1"),
            "--atlas",
            str(inputs / "per-grasp-motion-atlas-reserve-v2.json"),
            "--base",
            str(inputs / "regularized-image-supported-edge-motion-v1"),
            "--probe",
            str(inputs / "regularized-image-supported-edge-scene-probe-v1"),
            "--out",
            str(target),
        ],
        cwd=out / "methods",
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"),
        check=True,
    )
    import numpy as np

    with np.load(target / "robot-motion-proposal.npz", allow_pickle=False) as archive:
        actual = {name: archive[name] for name in archive.files}
    with np.load(
        case / "proposal/robot-motion-proposal.npz", allow_pickle=False
    ) as archive:
        expected = {name: archive[name] for name in archive.files}
    if set(actual) != set(expected):
        raise ValueError("Regenerated payload fields differ")
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
    receipt = {
        "schema": "v2d.regenerated-fitted-motion.v1",
        "all_payload_arrays_identical": True,
        "array_count": len(actual),
        "regenerated_payload_sha256": digest(target / "robot-motion-proposal.npz"),
        "frozen_payload_sha256": digest(case / "proposal/robot-motion-proposal.npz"),
        "fitted_archive_sha256": digest(args.fitted_archive),
        "upstream_reconstruction_or_fit_search_rerun": False,
        "physical_task_accepted": False,
    }
    (out / "reproduction-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
