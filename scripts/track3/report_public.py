#!/usr/bin/env python3
"""Build a browsable score and CAD overlay report for public Track 3 episodes."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from batch_public import roster


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def metric(metrics: dict, key: str, unit: str = "") -> str:
    value = metrics.get(key)
    return "—" if value is None else f"{value:.3f}{unit}"


def build(data: Path, root: Path) -> str:
    rows, cards = [], []
    for episode, names in roster(data):
        item = root / f"e{episode:03d}"
        observed = read_json(item / "scores_observed.json")
        inferred = read_json(item / "scores_inferred.json")
        masks = read_json(item / "masks/report.json")
        score = observed.get("mean_metrics", {})
        fill = inferred.get("mean_metrics", {})
        report = observed.get("episodes", [{}])[0]
        coverage = report.get("raw_observed_coverage_by_object", {})
        names_text = ", ".join(names)
        color_video = item / "cad_mesh_overlay_color.mp4"
        mesh_video = item / "cad_mesh_overlay.mp4"
        video = next((path for path in (color_video, mesh_video, item / "overlay.mp4")
                      if path.is_file()), item / "overlay.mp4")
        link = f'<a href="#episode-{episode}">view CAD mesh</a>' if (
            color_video.is_file() or mesh_video.is_file()) else (
            f'<a href="#episode-{episode}">view outline</a>' if video.is_file() else "pending")
        rows.append(f"<tr><td>{episode}</td><td>{html.escape(names_text)}</td>"
                    f"<td>{metric(score, 'add_auc')}</td>"
                    f"<td>{metric(score, 'maniptrans_sr')}</td>"
                    f"<td>{metric(score, 'spider_sr')}</td>"
                    f"<td>{metric(score, 'rpe_cm', ' cm')}</td>"
                    f"<td>{metric(score, 'mppe_cm', ' cm')}</td>"
                    f"<td>{link}</td></tr>")
        mask_text = ", ".join(f"{html.escape(k)} {v}/{masks.get('frames', '?')}"
                              for k, v in masks.get("mask_counts", {}).items())
        cover_text = ", ".join(f"{html.escape(k)} {v:.1%}" for k, v in coverage.items())
        video_html = (f'<video controls preload="none" src="e{episode:03d}/{video.name}"></video>'
                      if video.is_file() else "<p>Overlay pending.</p>")
        if color_video.is_file() and mesh_video.is_file():
            video_html += (f'<p><a href="e{episode:03d}/cad_mesh_overlay.mp4">'
                           'View natural CAD materials</a></p>')
        if (color_video.is_file() or mesh_video.is_file()) and (item / "overlay.mp4").is_file():
            video_html += (f'<p><a href="e{episode:03d}/overlay.mp4">'
                           'View outline diagnostic</a></p>')
        cards.append(f'<section id="episode-{episode}"><h2>Episode {episode}: '
                     f'{html.escape(names_text)}</h2>{video_html}'
                     f'<p>Observed: AUC {metric(score, "add_auc")}, '
                     f'MP-SR {metric(score, "maniptrans_sr")}, '
                     f'SP-SR {metric(score, "spider_sr")}, '
                     f'RPE {metric(score, "rpe_cm", " cm")}, '
                     f'MPPE {metric(score, "mppe_cm", " cm")}.</p>'
                     f'<p>Inferred gaps: AUC {metric(fill, "add_auc")}, '
                     f'MPPE {metric(fill, "mppe_cm", " cm")}. '
                     f'Masks: {mask_text or "pending"}. '
                     f'Observed pose coverage: {cover_text or "pending"}.</p>'
                     f'<p><a href="e{episode:03d}/scores_observed.json">Observed score JSON</a> · '
                     f'<a href="e{episode:03d}/scores_inferred.json">Inferred score JSON</a> · '
                     f'<a href="e{episode:03d}/masks/report.json">Mask report</a></p></section>')
    # Put the user's priority clip first in the video gallery.
    cards.sort(key=lambda card: 0 if 'id="episode-0"' in card else 1)
    return '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Track 3 public CAD tracking</title>
<style>
body{font:16px/1.5 system-ui,sans-serif;background:#f5f6fa;color:#202536;max-width:1200px;margin:auto;padding:24px}
h1,h2{line-height:1.2}p{max-width:850px}a{color:#144ca2}table{border-collapse:collapse;background:white;width:100%}
th,td{border-bottom:1px solid #ddd;padding:8px;text-align:left}th{position:sticky;top:0;background:#e9eef9}
.scroll{overflow-x:auto}section{background:white;padding:18px;margin:20px 0;border-radius:8px}
video{display:block;width:100%;max-width:800px}small{color:#50566a}
</style><h1>Track 3 public CAD tracking</h1>
<p>CAD outlines are drawn on rectified episode video. The table reports reconstruction diagnostics across
all public episodes. AUC, MP-SR, and SP-SR are higher when better; RPE and MPPE are in centimeters and
lower when better. A dash means the run is still pending or failed; it is not a zero score.</p>
<p>Camera and object predictions use video, calibration, object names, and public CAD. Public object poses
are used by the scorer only. The scorer aligns one constant transform for each episode. Inferred gaps
are scored separately from directly observed poses. This does not measure policy rollout success.</p>
<div class="scroll"><table><thead><tr><th>Episode</th><th>Objects</th><th>AUC</th><th>MP-SR</th>
<th>SP-SR</th><th>RPE</th><th>MPPE</th><th>Overlay</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table></div>
<h2>All episode overlays</h2>'''+''.join(cards)+'''</html>'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    (args.root / "index.html").write_text(build(args.data, args.root))
    print(args.root / "index.html")


if __name__ == "__main__":
    main()
