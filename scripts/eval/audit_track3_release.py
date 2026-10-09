#!/usr/bin/env python3
"""Audit what Track 3 actually releases, and what the scored roster requires.

Answers three questions that are easy to get wrong and expensive to discover late:

1. Which modalities does each split expose? ``track_3/evaluation`` is **RGB only** — no
   depth, no masks, no metadata, no reference. Anything in the pipeline that needs depth
   or masks has to be re-derived from video for the scored episodes.
2. Do the released videos cover every scored ``(episode, frame)`` in
   ``data/track_3_sample_submission.csv``? (They do, exactly — which means video frame
   index *is* the scored ``frame_index``, with no offset.)
3. Does every scored ``(episode, object)`` have released geometry for CD-O?
   ``wooden_spoon`` does not, which blocks the whole mesh submission because
   ``tools/pack_reconstruction.py`` requires all 36 pairs.

Usage::

    python scripts/eval/audit_track3_release.py --kit-root /path/to/v2d_submission_kit
    python scripts/eval/audit_track3_release.py --kit-root ... --hf   # also query the Hub
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
import subprocess
from pathlib import Path

REPO = "nvidia/video_to_data_challenge"


def roster(sample: Path):
    frames = collections.defaultdict(set)
    slots = collections.defaultdict(set)
    with open(sample) as fh:
        for row in csv.DictReader(fh):
            _, ep, world, frame, obj = row["row_id"].split("/")
            frames[int(ep[1:])].add(int(frame[1:]))
            slots[int(ep[1:])].add(int(obj[1:]))
    return frames, slots


def probe_modes(repo_root: Path):
    api_root = repo_root / "track_3"
    if not api_root.is_dir():
        return {}
    out = {}
    for split in sorted(p for p in api_root.iterdir() if p.is_dir()):
        per = collections.Counter()
        for path in split.rglob("*"):
            if path.is_file():
                rel = path.relative_to(split)
                per["/".join(rel.parts[:2])] += 1
        out[split.name] = per
    return out


def count_video_frames(path: Path) -> int:
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
             "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=120)
        return int(res.stdout.strip().splitlines()[0])
    except Exception:                                    # noqa: BLE001
        return -1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kit-root", required=True, type=Path,
                    help="extracted v2d_submission_kit directory")
    ap.add_argument("--eval-videos", type=Path,
                    help="track_3/evaluation/videos/chunk-000 (default: <kit-root>/../../track_3/evaluation/...)")
    ap.add_argument("--hf", action="store_true",
                    help="also query the HuggingFace Hub for the authoritative file list")
    args = ap.parse_args()

    sample = args.kit_root / "data" / "track_3_sample_submission.csv"
    objects = json.loads((args.kit_root / "data" / "track_3_evaluation_objects.json")
                         .read_text())["episodes"]
    frames, slots = roster(sample)

    pairs = [(int(ep), o) for ep, objs in objects.items() for o in objs]
    print(f"scored episodes            : {len(objects)}")
    print(f"scored frames              : {sum(len(v) for v in frames.values())}")
    print(f"scored (episode, object)   : {len(pairs)}")
    multi = sum(1 for ep, objs in objects.items() if len(objs) > 1)
    print(f"multi-object episodes      : {multi} (single-object episodes score RPE = 0 by definition)")

    print("\n== modalities per split ==")
    modes = probe_modes(args.kit_root)
    if modes:
        for split, per in modes.items():
            print(f"  {split}:")
            for k, v in sorted(per.items()):
                print(f"     {v:>4} {k}")
    if args.hf:
        from huggingface_hub import HfApi
        info = HfApi().repo_info(REPO, repo_type="dataset", files_metadata=True)
        per = collections.defaultdict(collections.Counter)
        for s in info.siblings:
            p = s.rfilename.split("/")
            if p[0] == "track_3" and len(p) > 2:
                per[p[1]]["/".join(p[2:-1]) or "(top level)"] += 1
        for split in sorted(per):
            print(f"  [hub] {split}:")
            for k, v in sorted(per[split].items()):
                print(f"     {v:>4} {k}")
        print("\n  NOTE: evaluation exposes videos only. No depth, masks, meta or reference,")
        print("        so any depth/mask stage has to run on RGB for the scored episodes.")
        geo = [s.rfilename for s in info.siblings
               if s.rfilename.startswith("track_3/public/mesh/")]
        have = {p.split("/")[3] for p in geo}
        need = {o for _, o in pairs}
        print(f"\n== CD-O geometry coverage ==\n  mesh dirs: {len(have)}  needed: {len(need)}")
        miss = sorted(need - have)
        print(f"  missing: {miss}")
        for m in miss:
            n = sum(1 for _, o in pairs if o == m)
            print(f"    {m}: {n} of {len(pairs)} scored pairs "
                  f"({n / len(pairs):.0%}) -- pack_reconstruction.py needs ALL pairs")

    if args.eval_videos and args.eval_videos.is_dir():
        print("\n== released video coverage vs scored frames ==")
        print(f"  {'ep':>4} {'scored':>7} {'cam_a':>7} {'cam_b':>7} {'cam_c':>7}")
        bad = []
        for ep in sorted(frames):
            counts = {}
            for d in sorted(os.listdir(args.eval_videos)):
                f = args.eval_videos / d / f"episode_{ep:06d}.mp4"
                if f.exists():
                    counts[d.rsplit("_", 1)[-1]] = count_video_frames(f)
            a = counts.get("a", -1)
            ok = a == len(frames[ep])
            if not ok:
                bad.append(ep)
            print(f"  {ep:>4} {len(frames[ep]):>7} {a:>7} "
                  f"{counts.get('b', -1):>7} {counts.get('c', -1):>7}"
                  f"{'' if ok else '   MISMATCH'}")
        print(f"\n  mismatches: {bad if bad else 'none'}")
        print("  exact match => video frame index IS the scored frame_index (no offset)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())