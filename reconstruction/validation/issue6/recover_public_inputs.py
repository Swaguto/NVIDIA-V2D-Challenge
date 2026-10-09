"""Recover the pinned full public split; verify every file against upstream IDs."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request


def hashes(path):
    sha256 = hashlib.sha256()
    sha1 = hashlib.sha1()
    sha1.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            sha256.update(block)
            sha1.update(block)
    return sha256.hexdigest(), sha1.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    if inventory["pagination_link"]:
        raise ValueError("Inventory is paginated; complete it before recovery")
    revision = inventory["revision"]
    root = args.root.resolve()
    items = [item for item in inventory["items"] if item["type"] == "file"]
    videos = [item for item in items if item["path"].endswith(".mp4")]
    if len(videos) != 60 or len({Path(x["path"]).stem for x in videos}) != 20:
        raise ValueError("Expected 20 episodes with all three cameras")

    def recover(item):
        relative = Path(item["path"]).relative_to("track_3/public")
        target = root / relative
        if not target.resolve().is_relative_to(root):
            raise ValueError("Input path escaped the dataset root")
        target.parent.mkdir(parents=True, exist_ok=True)
        reused = target.is_file()
        candidate = target if reused else target.with_suffix(target.suffix + ".partial")
        if not reused:
            url = f"https://huggingface.co/datasets/nvidia/video_to_data_challenge/resolve/{revision}/{item['path']}"
            with urllib.request.urlopen(url, timeout=90) as source, candidate.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
        if candidate.stat().st_size != item["size"]:
            raise ValueError(f"Byte count differs: {relative}")
        digest, blob = hashes(candidate)
        upstream = item.get("lfs", {}).get("oid")
        if (upstream and digest != upstream) or (not upstream and blob != item["oid"]):
            raise ValueError(f"Upstream content identity differs: {relative}")
        if not reused:
            candidate.replace(target)
        print(json.dumps({"file": relative.as_posix(), "bytes": item["size"], "reused": reused}), flush=True)
        return {"path": relative.as_posix(), "bytes": item["size"], "sha256": digest,
                "upstream_sha256": upstream, "git_blob_sha1": item["oid"] if not upstream else None}

    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in as_completed([pool.submit(recover, item) for item in items]):
            records.append(result.result())
    receipt = {"dataset": "nvidia/video_to_data_challenge", "revision": revision,
               "local_root": str(root), "episode_ids": sorted(int(Path(x["path"]).stem.split("_")[-1]) for x in videos if "ego_cam_c" in x["path"]),
               "video_files": len(videos), "files": sorted(records, key=lambda x: x["path"]),
               "reference_tables_downloaded": True, "reference_object_poses_used_in_inference": False,
               "scope": "Verified input reachability; not inference or episode acceptance"}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": "complete", "files": len(records), "bytes": sum(x["bytes"] for x in records), "receipt": str(args.receipt)}), flush=True)


if __name__ == "__main__":
    main()
