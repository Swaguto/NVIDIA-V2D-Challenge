"""Diagnostic only: reuse the GitHub smoother without importing its GPU tracker.

No organizer reference is accepted by this program. Only contiguous observed
source poses are smoothed, in the original optical-camera frame. The result is
a separate proposal; the source bundle and production motion are untouched.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "smoother-source", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    payload = args.bundle / "episode.npz"
    assert sha(payload) == manifest["payload"]["sha256"]
    source = np.load(payload, allow_pickle=False)
    if (
        sha(args.smoother_source)
        != "b06fff42a2fa7835e5461ab6e77fa59507c67e2c8239e30fbf1df4f40453487d"
    ):
        raise ValueError("Smoother source differs from the measured GitHub revision")
    tree = ast.parse(args.smoother_source.read_text(encoding="utf-8-sig"))
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in ("_solve_gain", "_eskf_rts")
    ]
    assert len(nodes) == 2
    entry = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "run_ekf_smoothing"
    )
    defaults = dict(
        zip(
            [n.arg for n in entry.args.args][-len(entry.args.defaults) :],
            [ast.literal_eval(n) for n in entry.args.defaults],
        )
    )
    params = {
        k: defaults[k]
        for k in (
            "process_noise_xy",
            "process_noise_z",
            "process_noise_r",
            "measurement_noise_xy",
            "measurement_noise_z",
            "measurement_noise_r",
            "min_iou",
        )
    }
    scope = {"np": np, "Rotation": Rotation}
    module = ast.Module(body=nodes, type_ignores=[])
    # Only the two original core functions from the exact hash above are loaded.
    exec(compile(module, str(args.smoother_source), "exec"), scope)  # noqa: S102
    solve = scope["_eskf_rts"]
    constant = np.tile([0.2, -0.1, 0.6], (10, 1))
    rotations = [Rotation.from_rotvec([0.1, 0.02, -0.1])] * 10
    control_t, control_r = solve(constant, rotations, np.ones(10), **params)
    np.testing.assert_allclose(control_t, constant, rtol=0, atol=1e-12)
    assert max((r * rotations[0].inv()).magnitude() for r in control_r) < 1e-12
    raw = np.stack(
        [source[f"object_{i}_world"] for i in range(len(manifest["objects"]))], axis=1
    ).copy()
    proposal = raw.copy()
    camera = source["camera_c2w"]
    inverse_camera = np.linalg.inv(camera)
    records = []
    for obj, item in enumerate(manifest["objects"]):
        observed = np.array(
            [
                manifest["object_observation_provenance"][item["name"]][str(int(f))]
                == "observed"
                for f in source["frame_indices"]
            ]
        )
        rows = np.flatnonzero(observed)
        breaks = np.flatnonzero(np.diff(rows) != 1) + 1
        for run in np.split(rows, breaks):
            if len(run) < 2:
                continue
            optical = inverse_camera[run] @ raw[run, obj]
            u, _, vt = np.linalg.svd(optical[:, :3, :3])
            optical[:, :3, :3] = u @ vt
            translations, rotations = solve(
                optical[:, :3, 3],
                [Rotation.from_matrix(m) for m in optical[:, :3, :3]],
                np.ones(len(run)),
                **params,
            )
            optical[:, :3, 3] = translations
            optical[:, :3, :3] = np.stack([r.as_matrix() for r in rotations])
            proposal[run, obj] = camera[run] @ optical
        delta = np.linalg.norm(proposal[:, obj, :3, 3] - raw[:, obj, :3, 3], axis=1)
        records.append(
            {
                "name": item["name"],
                "observed_frames": int(observed.sum()),
                "world_translation_change_m": {
                    "median": float(np.median(delta)),
                    "p90": float(np.quantile(delta, 0.9)),
                    "maximum": float(delta.max()),
                },
            }
        )
    args.out.mkdir(parents=True)
    output = args.out / "object-pose-proposal.npz"
    np.savez_compressed(
        output, object_world=proposal, frame_indices=source["frame_indices"]
    )
    report = {
        "schema": "v2d.existing-object-smoother-diagnostic.v1",
        "proposal_sha256": sha(output),
        "bundle_sha256": sha(payload),
        "source_sha256": sha(Path(__file__)),
        "existing_smoother_sha256": sha(args.smoother_source),
        "existing_core_ast_sha256": hashlib.sha256(
            ast.dump(module).encode()
        ).hexdigest(),
        "parameters_from_existing_function_defaults": params,
        "uniform_IoU_no_quality_scores_available": True,
        "coordinate_frame": "per-frame original optical camera",
        "only_contiguous_observed_runs_modified": True,
        "constant_pose_control_passed": True,
        "records": records,
        "GT_used_for_fitting": False,
        "source_or_production_motion_modified": False,
        "physical_task_accepted": False,
    }
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
