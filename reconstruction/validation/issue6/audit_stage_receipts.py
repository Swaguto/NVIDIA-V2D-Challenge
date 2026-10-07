"""Independently verify six-stage locks, source hashes and every output hash."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path

STAGES = ("validate", "hands", "depth", "align", "contacts", "report")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True, help="Direct run root containing eNNN"
    )
    parser.add_argument("--episodes", required=True, help="Comma-separated episode IDs")
    parser.add_argument("--modules-dir", type=Path, required=True)
    parser.add_argument("--verify-inputs", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cache = {}

    def digest(path):
        metadata = path.stat()
        key = (
            path.resolve(),
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
        if key not in cache:
            value = hashlib.sha256()
            with path.open("rb") as source:
                for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                    value.update(block)
            after = path.stat()
            if (after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
            ):
                raise ValueError(f"File changed during audit: {path}")
            cache[key] = value.hexdigest()
        return cache[key]

    results = []
    for episode in [int(value) for value in args.episodes.split(",")]:
        root = args.root / f"e{episode:03d}"
        result = {"episode": episode, "status": "pending"}
        missing = [
            stage for stage in STAGES if not (root / stage / "receipt.json").is_file()
        ]
        if not missing:
            try:
                lock = read(root / "inputs.lock.json")
                config_hash = hashlib.sha256(
                    json.dumps(lock, sort_keys=True, allow_nan=False).encode()
                ).hexdigest()
                if lock["manifest"]["episode"] != episode:
                    raise ValueError("Episode differs from locked manifest")
                for relative, expected in lock["implementation"].items():
                    source = (args.modules_dir / relative).resolve()
                    if (
                        not source.is_relative_to(args.modules_dir.resolve())
                        or digest(source) != expected
                    ):
                        raise ValueError(f"Implementation differs: {relative}")
                input_count = 0
                if args.verify_inputs:
                    for filename, expected in lock["inputs"].items():
                        source = Path(filename)
                        if expected == "MISSING":
                            if source.exists():
                                raise ValueError("Previously missing input now exists")
                        elif not source.is_file() or digest(source) != expected:
                            raise ValueError(f"Input differs: {source.name}")
                        input_count += 1
                stages = {}
                for stage in STAGES:
                    folder = root / stage
                    receipt = read(folder / "receipt.json")
                    actual = {
                        path.relative_to(folder).as_posix(): digest(path)
                        for path in sorted(folder.rglob("*"))
                        if path.is_file() and path.name != "receipt.json"
                    }
                    if (
                        receipt.get("stage") != stage
                        or receipt.get("status") != "complete"
                    ):
                        raise ValueError(f"Incomplete stage: {stage}")
                    if (
                        receipt.get("config_sha256") != config_hash
                        or receipt.get("outputs") != actual
                    ):
                        raise ValueError(f"Modified lock/output in stage: {stage}")
                    stages[stage] = {
                        "files": len(actual),
                        "receipt_sha256": digest(folder / "receipt.json"),
                    }
                result.update(
                    status="passed",
                    config_sha256=config_hash,
                    stages=stages,
                    implementation_files=len(lock["implementation"]),
                    verified_inputs=input_count,
                )
            except (KeyError, ValueError, OSError) as exc:
                result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        else:
            result["missing_stages"] = missing
        results.append(result)
        print(json.dumps({"episode": episode, "status": result["status"]}), flush=True)
    audit = {
        "verified_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "inputs_rehashed": args.verify_inputs,
        "episodes": results,
        "scope": "Content integrity and completed stages; not visual acceptance or contact accuracy",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if any(item["status"] == "failed" for item in results) or (
        args.require_complete and any(item["status"] != "passed" for item in results)
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
