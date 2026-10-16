"""Package a local, hash-verified coupled physics video and measured grasp summary."""

import argparse
import hashlib
import html
import json
from pathlib import Path
import subprocess

from audit_coupled_grip import summarize


def package(run, out):
    audit = summarize(run)
    report = json.loads((run / "report.json").read_text())
    source_video = run / "supported-task.mp4"
    if hashlib.sha256(source_video.read_bytes()).hexdigest() != report["video_sha256"]:
        raise ValueError("Original coupled video changed")
    out.mkdir(parents=True, exist_ok=False)
    video = out / "physics.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(source_video),
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-profile:v",
            "main",
            "-level:v",
            "4.0",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video),
        ],
        check=True,
    )
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames,duration",
                "-of",
                "json",
                str(video),
            ]
        )
    )
    if int(probe["streams"][0]["nb_read_frames"]) != report["rendered_video_frames"]:
        raise ValueError("Decoded preview frame count differs")
    receipt = dict(
        source_video_sha256=report["video_sha256"],
        preview_video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
        decoded_video=probe,
        all_timestamps_and_frames_preserved=True,
        original_source_frames=report["source_frames"],
        inferred_postroll_frames=report["rendered_video_frames"]
        - report["source_frames"],
    )
    (out / "grasp-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (out / "video-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    content = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Coupled Dex3 physics readback</title><style>body{background:#0d141e;color:#e6edf6;font:16px system-ui;margin:auto;max-width:1280px;padding:25px}p{line-height:1.6;color:#c4cfdf}video{width:100%}table{width:100%;border-collapse:collapse}td,th{text-align:left;border-bottom:1px solid #344355;padding:10px}a{color:#8ec7ff}</style><h1>Coupled Dex3 physics readback</h1><p>Controller diagnostic with dynamic objects and two floating hands. No learned policy or official evaluation is claimed. Object errors use original inferred source trajectories, including any remaining reconstruction error.</p><video controls playsinline preload="metadata" src="physics.mp4"></video>"""
    content += f"<p>{report['source_frames']} original source frames, followed by {receipt['inferred_postroll_frames']} labelled inferred release/settling frames. Video uses the source clock; physics runs at {report['source_motion_speed']} source speed. No frame is dropped.</p>"
    content += "<h2>Original-source object tracking</h2><p>Observed steps only; postroll excluded. These are local diagnostic metrics.</p><table><tr><th>Object</th><th>Mean</th><th>90th percentile</th><th>Maximum</th></tr>"
    for row in audit["objects"]:
        content += f"<tr><td>{row['object']}</td><td>{row['position_mean_m'] * 1000:.1f} mm</td><td>{row['position_p90_m'] * 1000:.1f} mm</td><td>{row['position_max_m'] * 1000:.1f} mm</td></tr>"
    content += "</table><h2>Grasp readback</h2><p>Position retention is separate from rotation, contact coverage and task success.</p><table><tr><th>Grasp</th><th>Maximum relative position drift</th><th>Maximum relative rotation drift</th><th>Contact gate active</th></tr>"
    for row in audit["grasps"]:
        content += f"<tr><td>{html.escape(row['sample'])}</td><td>{row['maximum_relative_position_drift_m'] * 1000:.1f} mm</td><td>{row['maximum_relative_rotation_drift_deg']:.1f}°</td><td>{row['contact_gate_coverage_fraction'] * 100:.1f}%</td></tr>"
    content += '</table><p>The contact gate requires declared measured digits, with a 30 ms dropout hold. Planned pad force is not measured pad force. Initial clearance correction and any ongoing contact correction remain experimental; objects receive no external wrench, attachment or pose writes after initialization.</p><p><a href="grasp-audit.json">Hash-bound measured summary</a> · <a href="video-receipt.json">Video provenance</a></p></html>'
    (out / "index.html").write_text(content)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.run, args.out), indent=2))
