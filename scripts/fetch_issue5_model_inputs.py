"""Download only pinned model data into the isolated issue-5 diagnostic cache."""
from pathlib import Path
import hashlib
import json
import urllib.request

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--max-bytes", type=int, default=2_000_000_000)
args = parser.parse_args()
HERE = Path(__file__).resolve().parents[1] / "docs/team/issue5/evidence"
ROOT = args.output
expected_records = json.loads((HERE / "episode39_models.json").read_text())
if sum(r["bytes"] for r in expected_records) > args.max_bytes:
    raise ValueError("model inputs exceed download byte bound")
specs = {}
for row in expected_records:
    spec = specs.setdefault(row["model"], {"sha":row["revision"],"files":[]})
    if spec["sha"] != row["revision"]:
        raise ValueError("inconsistent pinned model revisions")
    spec["files"].append({"path":row["file"],"bytes":row["bytes"],"lfs":{"sha256":row["sha256"]}})
records = []
for model, spec in specs.items():
    directory = ROOT / model.replace("/", "--") / spec["sha"]
    directory.mkdir(parents=True, exist_ok=True)
    for item in spec["files"]:
        relative = Path(item["path"])
        target = directory / relative
        if not target.resolve().is_relative_to(directory.resolve()):
            raise ValueError("model file leaves destination")
        expected = item["lfs"]["sha256"] if item["lfs"] else None
        if not target.exists():
            url = f"https://huggingface.co/{model}/resolve/{spec['sha']}/{relative.as_posix()}"
            partial = target.with_suffix(target.suffix + ".part")
            count, digest = 0, hashlib.sha256()
            with urllib.request.urlopen(url, timeout=60) as response, partial.open("xb") as output:
                while block := response.read(1024 * 1024):
                    count += len(block)
                    if count > item["bytes"]:
                        raise ValueError("model response exceeds advertised size")
                    digest.update(block)
                    output.write(block)
            if count != item["bytes"] or expected and digest.hexdigest() != expected:
                raise ValueError("model download differs from pinned bytes/hash")
            partial.rename(target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if target.stat().st_size != item["bytes"] or expected and digest != expected:
            raise ValueError("existing model differs from pinned bytes/hash")
        records.append({"model": model, "revision": spec["sha"], "file": item["path"],
                        "bytes": item["bytes"], "sha256": digest})
        print(f"verified {model}/{item['path']}: {item['bytes']} bytes", flush=True)
(ROOT / "model_download_manifest.json").write_text(json.dumps(records, indent=2))
print("Pinned model data download completed.", flush=True)
