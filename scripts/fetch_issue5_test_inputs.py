"""Fetch the hash-pinned official participant kit for issue-5 CPU parity checks."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

URL = "https://raw.githubusercontent.com/nvidia-isaac/video_to_data/main/docs/v2d_challenge/assets/v2d_submission_kit.zip"
SHA256 = "b1f8f703c2e772684a57d048cf28f40c7b9b54cf7a61aff6917b46d826915e1f"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="fresh external cache directory")
    parser.add_argument("--kit-zip", type=Path, help="reuse a local archive after checking the same hash")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("test-input output must be a fresh directory")
    if args.kit_zip:
        content = args.kit_zip.read_bytes()
    else:
        with urllib.request.urlopen(URL, timeout=60) as response:
            content = response.read(5_000_001)
        if len(content) > 5_000_000:
            raise ValueError("kit exceeds the expected download bound")
    if hashlib.sha256(content).hexdigest() != SHA256:
        raise ValueError("official kit hash changed; inspect the revision before using it")
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / "v2d_submission_kit.zip"
    archive.write_bytes(content)
    with zipfile.ZipFile(archive) as kit:
        for item in kit.infolist():
            if not (args.output / item.filename).resolve().is_relative_to(args.output.resolve()):
                raise ValueError("kit member leaves output root")
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("kit symlinks are unsupported")
        kit.extractall(args.output)
    root = args.output / "v2d_submission_kit"
    if not (root / "metric_code/track_3/AUC.py").is_file():
        raise ValueError("expected participant-kit structure is missing")
    manifest = {"source": URL, "sha256": SHA256, "purpose": "synthetic parity checks, no rollout",
                "kit_root": str(root.resolve())}
    (args.output / "test_input_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Set V2D_SUBMISSION_KIT to {root.resolve()}")


if __name__ == "__main__":
    main()
