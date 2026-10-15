"""Build a local video review from completed, hash-verified physics experiments."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reproduction"))
from audit_wrist_run import audit


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(experiments, out):
    if out.exists():
        raise FileExistsError("Use a new review directory")
    verified = []
    # Verify every source before creating the review; failed tracking is valid
    # evidence, while changed or incomplete artifacts cannot be displayed as checked.
    for label, stage in experiments:
        run = stage / "run/simulator"
        gate = json.loads((stage / "wrist-gate.json").read_text())
        recomputed = audit(run)
        metadata_matches = {k: v for k, v in gate.items() if k != "per_side"} == {
            k: v for k, v in recomputed.items() if k != "per_side"
        }
        # Scalar rotation summaries can round by a few ULPs between x86 and Arm.
        # Hashes, thresholds, failure lists and acceptance decisions stay exact.
        summaries_match = gate["per_side"].keys() == recomputed[
            "per_side"
        ].keys() and all(
            values.keys() == recomputed["per_side"][side].keys()
            and all(
                math.isclose(
                    value, recomputed["per_side"][side][key], abs_tol=1e-12, rel_tol=0
                )
                for key, value in values.items()
            )
            for side, values in gate["per_side"].items()
        )
        if not metadata_matches or not summaries_match:
            raise ValueError(f"Saved gate differs from current readback audit: {stage}")
        report = json.loads((run / "report.json").read_text())
        probe = json.loads((stage / "video-verification.json").read_text())
        if int(probe["streams"][0]["nb_read_frames"]) != report["source_frames"]:
            raise ValueError("Decoded video receipt has a different frame count")
        independent_path = stage / "independent-full-task-audit/report.json"
        independent = None
        if independent_path.exists():
            independent = json.loads(independent_path.read_text())
            if (
                independent["states_sha256"] != report["state_sha256"]
                or independent["video_sha256"] != report["video_sha256"]
            ):
                raise ValueError("Independent contact audit has different source bytes")
        verified.append((label, stage, run, gate, report, independent))
    out.mkdir(parents=True)
    cards, summaries = [], []
    for index, (label, stage, run, gate, report, independent) in enumerate(verified):
        folder = out / f"experiment-{index + 1}"
        folder.mkdir()
        for source in [
            stage / "wrist-gate.json",
            run / "report.json",
            stage / "invocation.json",
            stage / "video-verification.json",
            run / "supported-task.mp4",
        ]:
            shutil.copy2(source, folder / source.name)
        rows = "".join(
            f"<tr><td>{side}</td><td>{value['position_p95_m'] * 100:.2f}</td>"
            f"<td>{value['position_max_m'] * 100:.2f}</td>"
            f"<td>{value['rotation_p95_deg']:.2f}</td></tr>"
            for side, value in gate["per_side"].items()
        )
        contact = ""
        if independent:
            shutil.copy2(
                stage / "independent-full-task-audit/report.json",
                folder / "contact-audit.json",
            )
            contact = (
                "<p>Sampled source grasp cores with positive contact:</p><ul>"
                + "".join(
                    f"<li>{html.escape(item['interval'])}: {len(item['physical_contact_frames'])}/{item['source_definite_core_frames']}</li>"
                    for item in independent["source_grasp_comparison"]
                )
                + "</ul><p>Contact alone does not establish a grasp or object retention.</p>"
            )
        status = "PASS" if gate["controller_tracking_pass"] else "FAIL"
        scope = "full" if gate["complete_source_episode"] else "prefix"
        path = folder.name
        cards.append(
            f"<article><h2>{html.escape(label)}</h2><p>Wrist gate: <strong>{status}</strong> · "
            f"{report['source_frames']} source frames · {html.escape(gate['wrist_diagnostic']['controller'])}</p>"
            f'<video data-scope="{scope}" controls playsinline preload="metadata" src="{path}/supported-task.mp4"></video>'
            "<table><thead><tr><th>Hand</th><th>p95 position (cm)</th><th>Max position (cm)</th><th>p95 rotation (°)</th></tr></thead>"
            f"<tbody>{rows}</tbody></table><p>Physical task accepted: <strong>NO</strong>. Learned policy evaluated: <strong>NO</strong>.</p>"
            f'{contact}<p><a href="{path}/wrist-gate.json">Tracking gate</a> · <a href="{path}/report.json">Simulator report</a>'
            f' · <a href="{path}/invocation.json">Invocation</a></p></article>'
        )
        summaries.append(
            {
                "label": label,
                "stage": str(stage.resolve()),
                "gate_sha256": digest(stage / "wrist-gate.json"),
                "report_sha256": digest(run / "report.json"),
                "state_sha256": report["state_sha256"],
                "video_sha256": report["video_sha256"],
                "controller_tracking_pass": gate["controller_tracking_pass"],
                "physical_task_accepted": False,
            }
        )
    page = (
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dex3 physics validation</title><style>body{margin:0;background:#0c1016;color:#e5ecf4;font:16px system-ui}main{max-width:1500px;margin:auto;padding:24px}p,li{line-height:1.5;color:#c1ccda}a{color:#8ec7ff}.notice{border-left:4px solid #edbe68;padding:12px 18px;background:#211e17}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,590px),1fr));gap:20px}article{background:#151d28;border:1px solid #2a394a;border-radius:12px;padding:18px}video{width:100%;background:#080a0d}table{border-collapse:collapse;width:100%;font-size:14px;margin:16px 0}td,th{padding:8px;border-bottom:1px solid #344355;text-align:left}button{background:#284768;color:white;border:0;border-radius:6px;padding:10px;margin:0 8px 20px 0;cursor:pointer}h1{font-size:30px}h2{font-size:22px}</style><main><h1>Dex3 physics validation</h1>
<p class="notice">Gravity and contacts are enabled for the full runs. Objects receive no pose playback or external forces. These are controller diagnostics with zero policy residual. Passing wrist tracking does not accept a physical grasp or learned policy.</p>
<p>Playback follows source time; the physical reference runs at half speed. The support surface, object targets and robot retargeting remain inferred. Complete frame counts and contact evidence appear below.</p>
<p>Wrist limits are unchanged: per-hand p95 position ≤2 cm, maximum ≤10 cm; p95 rotation ≤10°, maximum ≤45°. Every recorded physics step, including startup, is included.</p>
<button onclick="document.querySelectorAll('video[data-scope=full]').forEach(v=>{v.currentTime=0;v.play()})">Play full runs together</button><button onclick="document.querySelectorAll('video').forEach(v=>v.pause())">Pause all</button>
<div class="grid">"""
        + "".join(cards)
        + '</div><p><a href="review-manifest.json">Hash-bound review sources</a></p></main></html>'
    )
    (out / "index.html").write_text(page)
    receipt = {
        "schema": "v2d.physics-review.v1",
        "experiments": summaries,
        "physical_task_accepted": False,
        "learned_policy_evaluated": False,
    }
    (out / "review-manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment", action="append", required=True, metavar="LABEL=STAGE_DIRECTORY"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    experiments = []
    for value in args.experiment:
        label, separator, path = value.partition("=")
        if not separator or not label or not path:
            parser.error("Each experiment must be LABEL=STAGE_DIRECTORY")
        experiments.append((label, Path(path)))
    print(json.dumps(build(experiments, args.out), indent=2))


if __name__ == "__main__":
    main()
