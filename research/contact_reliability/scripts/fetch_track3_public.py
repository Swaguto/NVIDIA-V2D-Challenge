"""Fetch small public pose tables and metadata at a pinned HF revision.

No videos, evaluation split, credentials, or simulator assets are downloaded.
Existing files must match the pinned bytes; they are never overwritten silently.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("data/track3"))
    args = p.parse_args()
    base = "https://huggingface.co"
    dataset = "nvidia/video_to_data_challenge"
    info = json.loads(fetch(f"{base}/api/datasets/{dataset}"))
    revision = info["sha"]
    tree = json.loads(fetch(f"{base}/api/datasets/{dataset}/tree/{revision}/track_3/public?recursive=true&limit=1000"))
    selected = [x for x in tree if x["type"] == "file" and
                (x["path"].startswith("track_3/public/data/") and x["path"].endswith(".parquet")
                 or x["path"].startswith("track_3/public/meta/"))]
    if sum(x["size"] for x in selected) > 20_000_000:
        raise ValueError("public metadata exceeded expected 20 MB bound")
    records = []
    for item in selected:
        relative = Path(item["path"]).relative_to("track_3")
        target = args.output / relative
        if not target.resolve().is_relative_to(args.output.resolve()):
            raise ValueError("unsafe dataset path")
        content = fetch(f"{base}/datasets/{dataset}/resolve/{revision}/{item['path']}")
        if target.exists() and target.read_bytes() != content:
            raise ValueError(f"existing file differs: {target}; choose a new output directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        records.append({"path": relative.as_posix(), "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest()})
    manifest = {"dataset": dataset, "revision": revision, "files": records,
                "scope": "public pose tables and metadata only; not training-ready references"}
    (args.output / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"revision": revision, "files": len(records), "bytes": sum(r['bytes'] for r in records)}))


if __name__ == "__main__":
    main()
