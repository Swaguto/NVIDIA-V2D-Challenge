"""Acquire hash-verified scored videos and shared assets, without pose labels.

Use a saved, complete Hugging Face tree at an immutable revision. Public pose
parquets are deliberately excluded. Reusing provided meshes does not establish
participant reconstruction provenance or submission eligibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath


def hashes(path):
    sha = hashlib.sha256()
    blob = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
            blob.update(block)
    return sha.hexdigest(), blob.hexdigest()


def select(inventory, roster, videos_only=False):
    if inventory.get("pagination_link"):
        raise ValueError("Dataset inventory must include every page")
    if not re.fullmatch(r"[0-9a-f]{40}", inventory["revision"]):
        raise ValueError("Pin an immutable dataset revision")
    selected, videos, paths = [], {}, set()
    for item in inventory["items"]:
        if item["type"] != "file":
            continue
        path = PurePosixPath(item["path"])
        if path.is_absolute() or ".." in path.parts or str(path) in paths:
            raise ValueError("Unsafe or duplicate upstream path")
        paths.add(str(path))
        if str(path).startswith("track_3/evaluation/"):
            match = re.fullmatch(
                r"track_3/evaluation/videos/chunk-000/observation.images."
                r"(ego_cam_[abc])/episode_(\d{6})\.mp4",
                str(path),
            )
            if not match or str(int(match[2])) not in roster:
                raise ValueError("Unexpected evaluation input")
            key = (int(match[2]), match[1])
            if key in videos:
                raise ValueError("Duplicate video")
            videos[key] = item
            selected.append(item)
        elif videos_only:
            if str(path) == "track_3/public/meta/camera_calibration.json":
                selected.append(item)
        elif any(
            str(path).startswith(f"track_3/public/{part}/")
            for part in ("meta", "mesh", "urdf")
        ) or str(path) in {"track_3/README.md", "track_3/v2d-challenge-track3.md"}:
            selected.append(item)
    expected = {
        (int(ep), cam)
        for ep in roster
        for cam in ("ego_cam_a", "ego_cam_b", "ego_cam_c")
    }
    if set(videos) != expected:
        raise ValueError("Every scored episode needs all three cameras")
    return selected


def verify(path, item):
    if path.stat().st_size != item["size"]:
        raise ValueError(f"Upstream byte count differs: {item['path']}")
    sha, blob = hashes(path)
    lfs = item.get("lfs", {}).get("oid")
    if (lfs and sha != lfs) or (not lfs and blob != item["oid"]):
        raise ValueError(f"Upstream content identity differs: {item['path']}")
    return sha


def acquire(inventory, roster, root, receipt, videos_only=False):
    items = select(inventory, roster, videos_only)
    if videos_only and not any(
        item["path"] == "track_3/public/meta/camera_calibration.json" for item in items
    ):
        raise ValueError("Video-only acquisition requires shared camera calibration")
    root = root.resolve()
    if receipt.exists():
        raise FileExistsError(
            "Use a fresh receipt; existing files are verified and reused"
        )

    def fetch(item):
        target = root / item["path"]
        if not target.resolve().is_relative_to(root):
            raise ValueError("Dataset destination escaped root")
        target.parent.mkdir(parents=True, exist_ok=True)
        reused = target.exists()
        candidate = target if reused else target.with_name(target.name + ".partial")
        if not reused:
            url = (
                "https://huggingface.co/datasets/nvidia/video_to_data_challenge/resolve/"
                + inventory["revision"]
                + "/"
                + item["path"]
            )
            with (
                urllib.request.urlopen(url, timeout=90) as stream,
                candidate.open("wb") as out,
            ):
                shutil.copyfileobj(stream, out, length=1024 * 1024)
        digest = verify(candidate, item)
        if not reused:
            candidate.replace(target)
        return {
            "path": item["path"],
            "size": item["size"],
            "sha256": digest,
            "reused": reused,
        }

    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(fetch, item) for item in items]):
            records.append(future.result())
            print(
                json.dumps(
                    {
                        "verified": len(records),
                        "total": len(items),
                        "path": records[-1]["path"],
                    }
                ),
                flush=True,
            )
    result = {
        "schema": "v2d.scored-input-acquisition.v1",
        "dataset": inventory["dataset"],
        "revision": inventory["revision"],
        "files": sorted(records, key=lambda r: r["path"]),
        "pose_labels_downloaded": False,
        "ground_truth_used_for_inference": False,
        "scored_episodes": sorted(map(int, roster)),
        "provided_meshes_are_not_claimed_as_reconstructions": True,
        "videos_only": videos_only,
        "provided_assets_downloaded": any(
            item["path"].startswith(("track_3/public/mesh/", "track_3/public/urdf/"))
            for item in items
        ),
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inventory", "roster", "root", "receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--videos-only",
        action="store_true",
        help="Acquire only the scored videos and shared camera calibration for inference",
    )
    args = parser.parse_args()
    result = acquire(
        json.loads(args.inventory.read_text()),
        json.loads(args.roster.read_text())["episodes"],
        args.root,
        args.receipt,
        args.videos_only,
    )
    print(json.dumps({"complete": True, "files": len(result["files"])}))
