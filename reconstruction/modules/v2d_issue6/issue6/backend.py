"""Adapters to shipped GPU runners. No downloads, model training, or provisioning."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from .manifest import read_json, real_file, write_json

MODULES = Path(__file__).resolve().parents[2]


def command(args, out, cwd=None):
    """Preserve exact argv and subprocess output, including failed runs."""
    with (out / "commands.jsonl").open("a", encoding="utf-8") as f:
        import json

        f.write(json.dumps([str(x) for x in args]) + "\n")
    with (out / "execution.log").open("a", encoding="utf-8") as log:
        subprocess.run(
            [str(x) for x in args],
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def fps(m):
    times = np.array([f["timestamp_s"] for f in m["frames"]])
    if len(times) < 2:
        raise ValueError("Video inference needs at least two timestamps")
    dt = np.diff(times)
    if not np.allclose(dt, np.median(dt), rtol=0.01, atol=1e-6):
        raise ValueError("Video backend requires regular sampling; preserve irregular data for import mode")
    return float(1 / np.median(dt))


def gpu_preflight(source, temporal=False, inference=True):
    if os.name != "posix" or not shutil.which("docker"):
        raise RuntimeError("GPU stages need the prepared Linux Docker host; CPU stages remain available")
    root = Path(source["mano_assets_root"])
    real_file(root / "models" / "MANO_RIGHT.pkl")
    if inference and not source.get("model_files"):
        raise ValueError("Declare model_files so model weights are checked and hashed")
    for p in source.get("model_files", []):
        real_file(p)
    if inference and not temporal:
        checkpoint = Path(source["weights_dir"]) / "_DATA" / "hamer_ckpts" / "checkpoints" / "hamer.ckpt"
        real_file(checkpoint)
    if temporal:
        bmc = Path(source["dynhamr_weights"]) / "BMC"
        if not list(bmc.glob("*.npy")):
            raise ValueError("Dyn-HaMR BMC assets are missing")
        real_file(Path(source["dynhamr_weights"]) / "models" / "MANO_RIGHT.pkl")


def _copy_records(source, out, frames, remap=False):
    mapping = {i: f["id"] for i, f in enumerate(frames)}
    allowed = {f["id"] for f in frames}
    count = 0
    for path in sorted(Path(source).glob("*/*.json")):
        r = read_json(real_file(path))
        idx = r["frame_idx"]
        if remap:
            if idx not in mapping:
                raise ValueError("Backend returned an unexpected frame")
            idx = mapping[idx]
        elif idx not in allowed:
            continue
        if not isinstance(r.get("is_right"), bool):
            raise ValueError("Hand record needs boolean is_right")
        if str(r["track_id"]) != path.parent.name or int(path.stem) != r["frame_idx"]:
            raise ValueError("Hand file/track/frame identifiers disagree")
        if remap:
            r["inference_frame_idx"] = r["frame_idx"]
        else:
            r.setdefault("inference_frame_idx", r["frame_idx"])
        r["frame_idx"] = idx
        # Do not alter source files. Explicit original-frame copies live in this run.
        write_json(out / "records" / str(r["track_id"]) / f"{idx:06d}.json", r)
        count += 1
    if count == 0:
        raise ValueError("No hand records in selected frames")


def _copy_masks(mask_root, track_path, out, frames, remap=False):
    """Canonical masks by handedness; refuse ambiguous multiple same-side tracks."""
    tracks = read_json(real_file(track_path))["tracks"]
    hand_tracks = [t for t in tracks if t.get("role", "hand") == "hand"]
    for frame_no, f in enumerate(frames):
        src_id = frame_no if remap else f["id"]
        for side in (False, True):
            paths = [
                Path(mask_root) / str(t["object_id"]) / f"{src_id:06d}.png"
                for t in hand_tracks
                if t["is_right"] is side
            ]
            paths = [p for p in paths if p.exists()]
            if len(paths) == 1:
                target = out / "masks" / ("right" if side else "left") / f"{f['id']:06d}.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(real_file(paths[0]), target)


def prepare_render_inputs(frames_dir, records_dir, frame_axis, destination):
    """Upstream mesh renderer indexes by enumeration, not image filename."""
    destination = Path(destination)
    image_dir, hand_dir = destination / "frames", destination / "records"
    image_dir.mkdir(parents=True)
    mapping = {f["id"]: n for n, f in enumerate(frame_axis)}
    for original, sequential in mapping.items():
        shutil.copy2(
            Path(frames_dir) / f"{original:06d}.png",
            image_dir / f"{sequential:06d}.png",
        )
    for p in Path(records_dir).glob("*/*.json"):
        r = read_json(p)
        original = r["frame_idx"]
        if original not in mapping:
            raise ValueError("Unexpected frame in mesh-render input")
        r["original_frame_idx"] = original
        r["frame_idx"] = mapping[original]
        write_json(hand_dir / p.parent.name / f"{mapping[original]:06d}.json", r)
    write_json(
        destination / "frame_map.json",
        [{"render_index": n, **f} for n, f in enumerate(frame_axis)],
    )
    return image_dir, hand_dir


def reconstruct(m, out):
    source = m["hand_source"]
    backend = source["backend"]
    if backend == "import":
        _copy_records(source["records_dir"], out, m["frames"])
        if source.get("masks_dir") and source.get("tracks_path"):
            _copy_masks(source["masks_dir"], source["tracks_path"], out, m["frames"])
        return
    gpu_preflight(source, backend == "dynhamr")
    command(
        [
            "docker",
            "image",
            "inspect",
            "v2d_hamer",
            "v2d_mediapipe",
            "v2d_sam2",
            "--format",
            "{{.Id}}",
        ],
        out,
    )
    command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        out,
    )
    rate = fps(m)
    pipeline = out / "mask_pipeline"
    frames = pipeline / "frames"
    frames.mkdir(parents=True)
    for n, f in enumerate(m["frames"]):
        with Image.open(real_file(f["color"])) as im:
            im.convert("RGB").save(frames / f"{n:06d}.png")
    video = out / "selected.mp4"
    command(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-framerate",
            rate,
            "-i",
            frames / "%06d.png",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            video,
        ],
        out,
    )
    # This runs only detection/SAM2. Intentionally omit --run_hamer, which would
    # also run MoGe and entangle inference with monocular depth alignment.
    command(
        [
            sys.executable,
            MODULES / "v2d_pipelines" / "run_hand_masks.py",
            "--video_path",
            video,
            "--output_dir",
            pipeline,
            "--reference_frame",
            [f["id"] for f in m["frames"]].index(m.get("settings", {}).get("reference_frame_id", m["frames"][0]["id"])),
            "--sam2_weights",
            source["sam2_weights"],
        ],
        out,
        MODULES.parent,
    )
    if backend == "hamer":
        raw = out / "native_hamer"
        command(
            [
                sys.executable,
                "-m",
                "v2d.hamer.docker.run_masks_to_hands",
                "--frames_dir",
                frames,
                "--masks_dir",
                pipeline / "masks",
                "--tracks_path",
                pipeline / "hand_tracks.json",
                "--output_dir",
                raw,
                "--weights_dir",
                source["weights_dir"],
            ],
            out,
        )
        command(
            [
                sys.executable,
                "-m",
                "v2d.hamer.docker.run_render_hands_video",
                "--frames_dir",
                frames,
                "--hamer_dir",
                raw,
                "--mano_assets_root",
                source["mano_assets_root"],
                "--output_path",
                out / "hand_baseline.mp4",
                "--fps",
                rate,
            ],
            out,
        )
    else:
        temporal = out / "native_dynhamr"
        command(
            [
                sys.executable,
                "-m",
                "v2d_ego_hand_reconstruction.docker.run_reconstruction",
                "--video_input",
                video,
                "--output_dir",
                temporal,
                "--weights_dir",
                source["dynhamr_weights"],
            ],
            out,
        )
        candidates = list(temporal.rglob("world_results.npz"))
        if len(candidates) != 1:
            raise ValueError("Dyn-HaMR needs exactly one world_results.npz; inspect native outputs")
        raw = out / "native_converted"
        intrinsics = out / "intrinsics.json"
        write_json(intrinsics, m["color_camera"])
        command(
            [
                sys.executable,
                "-m",
                "v2d.hand_alignment.docker.run_dynhamr_to_hamer_tracks",
                "--input_npz",
                candidates[0],
                "--output_dir",
                raw,
                "--intrinsics_path",
                intrinsics,
            ],
            out,
        )
    _copy_records(raw, out, m["frames"], remap=True)
    _copy_masks(pipeline / "masks", pipeline / "hand_tracks.json", out, m["frames"], remap=True)


def align_worker(m, run_root, out):
    source = m["hand_source"]
    gpu_preflight(source, inference=False)
    command(["docker", "image", "inspect", "v2d_hamer", "--format", "{{.Id}}"], out)
    # Read-only source and model mounts; one writable run directory. Existing
    # model container provides torch/manotorch/pyrender; no new image is built.
    command(
        [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "-e",
            "HOME=/tmp",
            "-e",
            "PYTHONPATH=/issue6",
            "-e",
            "PYOPENGL_PLATFORM=egl",
            "-v",
            f"{MODULES / 'v2d_issue6'}:/issue6:ro",
            "-v",
            f"{MODULES}:/modules:ro",
            "-v",
            f"{run_root}:/job",
            "-v",
            f"{Path(source['mano_assets_root']).resolve()}:/mano:ro",
            "v2d_hamer",
            "python",
            "-m",
            "issue6.worker",
            "--job",
            "/job",
        ],
        out,
    )
