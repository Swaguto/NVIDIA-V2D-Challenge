"""Create a browser-compatible derivative while preserving the static physics video."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe(path):
    return json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=codec_name,width,height,nb_read_frames:frame=best_effort_timestamp_time",
                "-of",
                "json",
                str(path),
            ],
            text=True,
        )
    )


def package(run: Path):
    report = json.loads((run / "report.json").read_text())
    source = run / "static-load.mp4"
    if digest(source) != report["video_sha256"]:
        raise ValueError("Original static video differs from the simulator report")
    output = run / "static-load-browser.mp4"
    receipt_path = run / "browser-preview.json"
    if output.exists() or receipt_path.exists():
        raise FileExistsError("Preview already exists; preserve the previous delivery")
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(source),
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(command, check=True)
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(output), "-f", "null", "-"],
        check=True,
    )
    original, preview = probe(source), probe(output)
    a, b = original["streams"][0], preview["streams"][0]
    first = [float(x["best_effort_timestamp_time"]) for x in original["frames"]]
    second = [float(x["best_effort_timestamp_time"]) for x in preview["frames"]]
    if (
        b["codec_name"] != "h264"
        or int(a["nb_read_frames"]) != 21
        or any(a[key] != b[key] for key in ["width", "height", "nb_read_frames"])
        or len(first) != 21
        or len(second) != 21
        or any(abs(x - y) > 1e-9 for x, y in zip(first, second, strict=True))
        or any(y <= x for x, y in zip(second, second[1:]))
    ):
        raise ValueError(
            "Browser derivative dimensions, frame count or timestamps differ"
        )
    receipt = {
        "schema": "v2d.static-browser-preview.v1",
        "original_video_sha256": digest(source),
        "preview_video_sha256": digest(output),
        "frames": 21,
        "codec": "h264",
        "frame_timestamps_preserved": True,
        "original_physics_video_preserved": True,
        "preview_is_transcode": True,
        "command": command,
        "packager_sha256": digest(Path(__file__)),
        "ffmpeg_version": subprocess.check_output(
            ["ffmpeg", "-version"], text=True
        ).splitlines()[0],
        "physical_task_accepted": False,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    print(json.dumps(package(parser.parse_args().run), indent=2))
