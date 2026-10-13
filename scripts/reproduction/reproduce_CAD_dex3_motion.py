"""Rebuild the original-CAD replay from verified saved fits and automatic repairs.

This materializes frozen fitted inputs and checks exact output bytes. Upstream
inference and optimizer searches are separate experiments, not rerun here.
"""

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from reproduce_frozen_dex3_replay import digest, restore_methods


def restore_case(key, case, kit, out, generated=None):
    """Rehydrate shared immutable files by verified content, preserving metadata."""
    manifest_path = kit / "expected" / key / "file-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source_manifest = json.loads((case / "file-manifest.json").read_text())
    lookup = {
        value: case / name for name, value in source_manifest["files_sha256"].items()
    }
    for path in kit.rglob("*"):
        if path.is_file():
            lookup[digest(path)] = path
    out.mkdir()
    for name, expected in manifest["files_sha256"].items():
        target = out / name
        if not target.resolve().is_relative_to(out.resolve()):
            raise ValueError("Case path escapes restored directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        source = lookup.get(expected)
        if generated and name in (
            "proposal/robot-motion-proposal.npz",
            "proposal/replay-knots.npz",
        ):
            source = generated / Path(name).name
        if source is None or digest(source) != expected:
            raise ValueError("Cannot reproduce exact case member: " + name)
        shutil.copyfile(source, target)
    shutil.copyfile(manifest_path, out / "file-manifest.json")
    # Report bytes above are the immutable measured fit input record. New
    # materializer reports remain alongside the regenerated payloads.
    return out


def run(python, methods, name, arguments):
    subprocess.run([str(python), str(methods / name), *map(str, arguments)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--fitted-archive", type=Path, required=True)
    parser.add_argument("--CAD-archive", dest="cad_archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Use a fresh output directory")
    snapshot = (
        Path(__file__).resolve().parents[2]
        / "docs/team/reproduction/dex3-motion-replay"
    )
    kit_manifest = snapshot / "CAD-fitted-method-kit-manifest-v1.json"
    expected_kit = json.loads(kit_manifest.read_text())
    case = args.case.resolve()
    if (
        digest(case / "file-manifest.json")
        != expected_kit["published_case_manifest_sha256"]
    ):
        raise ValueError("Use the verified original-CAD replay case")
    manifest = json.loads((case / "file-manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        path = case / name
        if not path.resolve().is_relative_to(case) or digest(path) != expected:
            raise ValueError("Case member hash/path mismatch: " + name)
    out = args.out.resolve()
    out.mkdir(parents=True)
    restore_methods(args.cad_archive, kit_manifest, out / "kit")
    restore_methods(
        args.fitted_archive, snapshot / "fitted-input-manifest.json", out / "inputs"
    )
    if args.check_only:
        print(
            "Case, raw methods and fitted input bytes verified; no reconstruction or geometry run."
        )
        return
    expected_runtime = {
        "numpy": "2.2.6",
        "scipy": "1.15.3",
        "pin": "3.7.0",
        "coal": "3.0.1",
        "pin-pink": "4.2.0",
        "qpsolvers": "4.9.0",
        "daqp": "0.10.3",
        "trimesh": "4.12.2",
    }
    if sys.platform != "linux" or platform.python_version() != "3.11.17":
        raise RuntimeError(
            "Regeneration requires the recorded Linux Python 3.11.17 runtime"
        )
    for name, expected in expected_runtime.items():
        if importlib.metadata.version(name) != expected:
            raise RuntimeError("Recorded geometry package mismatch: " + name)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    kit, inputs = out / "kit", out / "inputs"
    methods = kit / "methods"
    cases, generated = out / "cases", out / "generated"
    cases.mkdir()
    generated.mkdir()
    original = restore_case("original", case, kit, cases / "original")
    v5 = generated / "v5"
    run(
        sys.executable,
        methods,
        "build_CAD_contact_replay_v2.py",
        [
            "--case",
            original,
            "--inputs",
            inputs,
            "--candidates",
            kit / "fitted/static/report.json",
            "--out",
            v5,
        ],
    )
    c5 = restore_case("v5", case, kit, cases / "v5", v5)
    v6 = generated / "v6"
    run(
        sys.executable,
        methods,
        "build_CAD_envelope_replay_v6.py",
        [
            "--case",
            c5,
            "--original-case",
            original,
            "--search",
            kit / "fitted/transition-search",
            "--methods",
            methods,
            "--out",
            v6,
        ],
    )
    current = restore_case("v6", case, kit, cases / "v6", v6)
    for index, version in ((1, 8), (2, 9), (3, 10)):
        target = generated / f"v{version}"
        fit = kit / f"fitted/repair-{index}"
        run(
            sys.executable,
            methods,
            "materialize_saved_CAD_counterexample_v4.py",
            [
                "--case",
                current,
                "--original-case",
                original,
                "--counterexample",
                fit / "counterexample.json",
                "--fitted-search",
                fit / "search-report.json",
                "--methods",
                methods,
                "--out",
                target,
            ],
        )
        current = restore_case(f"v{version}", case, kit, cases / f"v{version}", target)
    import numpy as np

    counts = {}
    for filename in ("robot-motion-proposal.npz", "replay-knots.npz"):
        actual_path = target / filename
        expected_path = case / "proposal" / filename
        if digest(actual_path) != digest(expected_path):
            raise ValueError("Final regenerated bytes differ: " + filename)
        with (
            np.load(actual_path, allow_pickle=False) as actual,
            np.load(expected_path, allow_pickle=False) as expected,
        ):
            if actual.files != expected.files:
                raise ValueError("Regenerated payload field order differs")
            for name in actual.files:
                np.testing.assert_array_equal(
                    actual[name], expected[name], err_msg=name
                )
            counts[filename] = len(actual.files)
    receipt = {
        "schema": "v2d.original-CAD-fitted-motion-reproduction.v1",
        "producer_sha256": digest(Path(__file__)),
        "runtime": {
            "platform": sys.platform,
            "python": platform.python_version(),
            "packages": expected_runtime,
        },
        "CAD_method_kit_sha256": digest(args.cad_archive),
        "all_payload_and_knot_arrays_identical": True,
        "array_counts": counts,
        "regenerated_payload_sha256": digest(target / "robot-motion-proposal.npz"),
        "regenerated_knots_sha256": digest(target / "replay-knots.npz"),
        "source_robot_poses_unchanged_by_between_sample_repairs": True,
        "immutable_measured_reports_used_as_fitted_input_metadata": True,
        "new_materializer_reports_retained": True,
        "fitting_or_upstream_inference_rerun": False,
        "fresh_finite_counterexample_window_CAD_checks_passed": True,
        "continuous_geometry_rerun": False,
        "new_simulator_run": False,
        "physical_task_accepted": False,
    }
    (out / "reproduction-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
