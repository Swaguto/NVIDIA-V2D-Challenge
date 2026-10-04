"""Fetch a bounded public diagnostic subset at the existing GT snapshot revision."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default="5f68335f3acc802033d1e80728c1633197521de8")
    parser.add_argument("--episodes", default="0,2")
    parser.add_argument("--max-bytes", type=int, default=150_000_000)
    args = parser.parse_args()
    episodes = {int(e) for e in args.episodes.split(",")}
    base = "https://huggingface.co"
    dataset = "nvidia/video_to_data_challenge"
    request = f"{base}/api/datasets/{dataset}/tree/{args.revision}/track_3/public?recursive=true&limit=1000"
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.headers.get("Link"):
            raise ValueError("unexpected paginated dataset tree")
        tree = json.load(response)
    selected = [item for item in tree if item["type"] == "file" and
        ("/mesh/" in item["path"] and not "_collision" in item["path"] or
         any(item["path"].endswith(f"/episode_{episode:06d}.mp4") for episode in episodes)
         and any(f"observation.images.{cam}/" in item["path"] for cam in ("ego_cam_b", "ego_cam_c")))]
    if sum(item["size"] for item in selected) > args.max_bytes:
        raise ValueError("diagnostic subset exceeds byte bound")
    records = []
    for item in selected:
        relative = Path(item["path"]).relative_to("track_3/public")
        target = args.output / relative
        if not target.resolve().is_relative_to(args.output.resolve()):
            raise ValueError("dataset path leaves output root")
        expected = item["lfs"]["oid"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            partial = target.with_suffix(target.suffix + ".part")
            if partial.exists():
                raise ValueError(f"prior partial download requires inspection: {partial}")
            url = f"{base}/datasets/{dataset}/resolve/{args.revision}/{item['path']}"
            with urllib.request.urlopen(url, timeout=60) as response, partial.open("xb") as output:
                total, digest = 0, hashlib.sha256()
                while block := response.read(1024 * 1024):
                    total += len(block)
                    if total > item["size"]:
                        raise ValueError("response exceeds advertised size")
                    digest.update(block); output.write(block)
            if total != item["size"] or digest.hexdigest() != expected:
                raise ValueError(f"download hash/size mismatch: {relative}")
            partial.rename(target)
        if target.stat().st_size != item["size"] or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError(f"existing asset does not match pinned dataset: {relative}")
        records.append({"path": relative.as_posix(), "bytes": item["size"], "sha256": expected})
        print(f"verified {relative.as_posix()}", flush=True)
    manifest = {"dataset": dataset, "revision": args.revision, "files": records,
        "episodes": sorted(episodes),
        "scope": "public visual meshes and selected episode stereo RGB; no prepared observations or cameras"}
    (args.output / "asset_download_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"files": len(records), "bytes": sum(r["bytes"] for r in records)}))


if __name__ == "__main__":
    main()
