"""Infer calibrated full-episode human hands from a video-only source mount.

Both official scored episodes and separately declared public validation videos
are supported. Public pose tables must stay outside the common inference mount.
A surrounding compute lease must enforce backup and shutdown.
"""

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from run_scored_reconstruction import (
    PERCEPTION,
    image_digest,
    sha,
    validate_video_mount,
)


def validate(config):
    if type(config["max_seconds"]) is not int or not 0 < config["max_seconds"] <= 3600:
        raise ValueError("Hand inference needs a deadline of at most one hour")
    image_digest(config["perception_image"])
    if config["split"] not in {"evaluation", "public"}:
        raise ValueError("Declare scored evaluation or public validation split")
    root = Path(config["source_root"]).resolve()
    relative = Path(config["data_subdir"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Video view must stay within the source mount")
    data = root / relative
    if (
        not (data / "videos").is_dir()
        or not (data / "meta/camera_calibration.json").is_file()
    ):
        raise ValueError("Video/calibration view is incomplete")
    validate_video_mount(root, data)
    roster = json.loads(Path(config["roster"]).read_text())
    names = roster["episodes"][str(config["episode"])]
    if (
        not names
        or len(set(names)) != len(names)
        or not all(n.isidentifier() for n in names)
    ):
        raise ValueError("Invalid object slot names")
    if config["split"] == "public" and (
        roster.get("schema") != "v2d.public-video-validation-roster.v1"
        or roster.get("split") != "public"
    ):
        raise ValueError("Public inference requires its explicit video-only roster")
    if config["split"] == "evaluation" and not Path(config["kit"]).is_dir():
        raise ValueError("Scored inference requires the official kit")
    overlay = Path(config["overlay"])
    manifest = json.loads((overlay / "context-manifest.json").read_text())
    if manifest.get("schema") != "v2d.scored-perception-context.v1" or any(
        manifest.get(k) is not False
        for k in (
            "private_footage_included",
            "model_weights_included",
            "credentials_included",
        )
    ):
        raise ValueError("Expected a committed code-only perception context")
    for name, expected in manifest["files"].items():
        path = (overlay / name).resolve()
        if not path.is_relative_to(overlay.resolve()) or sha(path) != expected:
            raise ValueError("Committed perception source differs")
    for name in (
        "hand-runtime-v1/hamer",
        "hand-runtime-v1/hamer/_DATA/data/mano",
        "hand-runtime-v1/public",
    ):
        if not (Path(config["models"]) / name).is_dir():
            raise ValueError("Missing private hand model mount: " + name)
    return manifest["source_commit"]


def run(config):
    commit = validate(config)
    out = Path(config["out"]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name in ("inputs", "observations", "temporal-hands"):
        (out / name).mkdir()
    shutil.copyfile(config["roster"], out / "inputs/roster.json")
    (out / "inputs/run_native.py").write_text(
        "import runpy,sys\nfrom pathlib import Path\nscript=sys.argv.pop(1);sys.path.insert(0,str(Path(script).parent));runpy.run_path(script,run_name='__main__')\n"
    )
    ep, label = config["episode"], f"e{config['episode']:03d}"
    base = Path(config["models"])
    data = "/source/" + config["data_subdir"]
    daemon = ["docker"] + (
        ["--host", config["docker_host"]] if config.get("docker_host") else []
    )
    deadline = time.monotonic() + config["max_seconds"]
    receipt = {
        "schema": "v2d.full-episode-video-hand-inference.v1",
        "episode": ep,
        "split": config["split"],
        "source_commit": commit,
        "perception_image": config["perception_image"],
        "roster_sha256": sha(Path(config["roster"])),
        "stages": [],
        "reference_pose_labels_read": False,
        "provided_object_meshes_read": False,
        "physical_task_accepted": False,
    }

    def save():
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    def container(script, args, name, native=False, gpu=False, limit=600):
        cname = f"v2d-hands-{label}-{name}"
        cmd = daemon + [
            "run",
            "--rm",
            "--pull=never",
            "--name",
            cname,
            "--network",
            "none",
            "--shm-size",
            "8g",
        ]
        cmd += (
            ["--gpus", "all"]
            if gpu
            else ["--cpus", "4", "-e", "NVIDIA_VISIBLE_DEVICES=void"]
        )
        mounts = [
            (Path(config["overlay"]) / "code", "/workspace/pr32", True),
            (Path(config["source_root"]), "/source", True),
            (out, "/stage", False),
            (out / "observations", "/workspace/issue6-full-inputs", False),
            (out / "temporal-hands", "/workspace/temporal-hands", False),
            (base, "/models", True),
            (base / "hand-runtime-v1/hamer", "/workspace/hamer-weights", True),
            (base / "hand-runtime-v1/hamer/_DATA", "/workspace/Dyn-HaMR/_DATA", True),
            (
                base / "hand-runtime-v1/hamer/_DATA/data/mano",
                "/workspace/mano/models",
                True,
            ),
            (base / "hand-runtime-v1/public", "/workspace/pr32-models", True),
        ]
        if config["split"] == "evaluation":
            mounts.append((Path(config["kit"]), "/kit", True))
        for src, dst, ro in mounts:
            cmd += ["-v", str(src.resolve()) + ":" + dst + (":ro" if ro else "")]
        for k, v in {
            "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics",
            "HF_HOME": "/models/huggingface",
            "HF_HUB_OFFLINE": "1",
            "V2D_DINO_REVISION": "12bdfa3120f3e7ec7b434d90674b3396eccf88eb",
            "V2D_SAM2_REVISION": "665f8e2ad61cf5f53d65644ff27c8ee525124610",
            "V2D_CALIB": data + "/meta/camera_calibration.json",
            "OMP_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
        }.items():
            cmd += ["-e", k + "=" + v]
        if native:
            cmd += ["--entrypoint", "/workspace/hand-runtime/bin/python"]
        cmd += [config["perception_image"]]
        if native:
            cmd += ["/stage/inputs/run_native.py"]
        cmd += [script, *map(str, args)]
        remaining = int(deadline - time.monotonic() - 10)
        if remaining <= 0:
            raise TimeoutError("Hand inference deadline")
        start = time.monotonic()
        try:
            with (out / (name + ".log")).open("wb") as log:
                result = subprocess.run(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=min(limit, remaining),
                    check=False,
                )
        except subprocess.TimeoutExpired:
            subprocess.run(
                daemon + ["rm", "-f", cname],
                capture_output=True,
                timeout=30,
                check=False,
            )
            receipt["stages"].append({"name": name, "timed_out": True})
            save()
            raise
        stage = {
            "name": name,
            "returncode": result.returncode,
            "wall_seconds": time.monotonic() - start,
        }
        receipt["stages"].append(stage)
        save()
        print(json.dumps(stage), flush=True)
        result.check_returncode()

    try:
        if config.get("cameras"):
            camera = Path(config["cameras"]) / label / "camera"
            shutil.copytree(camera, out / "perception" / label / "camera")
            receipt["reused_camera_report_sha256"] = sha(camera / "report.json")
        else:
            container(
                "/workspace/pr32/scripts/track3/batch_public.py",
                [
                    "--phase",
                    "cameras",
                    "--data",
                    data,
                    "--roster",
                    "/stage/inputs/roster.json",
                    "--episodes",
                    ep,
                    "--root",
                    "/stage/perception",
                ],
                "cameras",
                limit=600,
            )
        contract = (
            ["--scored-kit", "/kit"]
            if config["split"] == "evaluation"
            else ["--video-roster", "/stage/inputs/roster.json"]
        )
        container(
            "-m",
            [
                "issue6.track3_adapter",
                "--episode",
                ep,
                "--data",
                data,
                "--results",
                "/stage/perception",
                "--camera-root",
                "/stage/perception",
                "--output",
                f"/stage/observations/{label}",
                "--hands-only",
                *contract,
            ],
            "observations",
            limit=300,
        )
        container(
            "/workspace/pr32/scripts/track3/export_optimizer_cameras.py",
            [
                "--manifest",
                f"/stage/observations/{label}/manifest.json",
                "--out",
                f"/workspace/temporal-hands/{label}/cameras",
            ],
            "optimizer-cameras",
            limit=90,
        )
        for name in ("enhanced_hands", "full_hand_masks", "run_temporal_fit"):
            container(
                "/workspace/pr32/reconstruction/validation/issue6/" + name + ".py",
                ["--episode", ep],
                name,
                native=True,
                gpu=True,
                limit=900,
            )
        fit = out / f"temporal-hands/{label}"
        receipt["fit_receipt_sha256"] = sha(fit / "fit_receipt.json")
        receipt["observation_gate_sha256"] = sha(fit / "observation_gate.json")
        receipt["hand_fit_exported"] = True
    except BaseException as error:
        receipt["error"] = repr(error)
        raise
    finally:
        save()
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config.setdefault("perception_image", PERCEPTION)
    run(config)
