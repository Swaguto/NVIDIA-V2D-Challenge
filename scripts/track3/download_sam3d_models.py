"""Acquire the pinned scored SAM3D model cache outside Git and Docker images.

Authenticate with the official `hf auth login` locally. No credential is accepted
as a command argument, stored in a receipt, or sent to the auxiliary hosts.
The team must separately have accepted the gated model's license/access terms.
"""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

DEFAULT_LOCK = (
    Path(__file__).resolve().parents[2] / "docker/track3-scored/sam3d-models.lock.json"
)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path, record):
    if (
        not path.is_file()
        or ("size" in record and path.stat().st_size != record["size"])
        or sha(path) != record["sha256"]
    ):
        raise ValueError(f"Missing or changed pinned model file: {path.name}")


def fetch_url(url, path, record, offline):
    if path.exists():
        verify(path, record)
        return
    if offline:
        raise FileNotFoundError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    # URL requests never receive the Hugging Face token.
    urllib.request.urlretrieve(url, partial)
    verify(partial, record)
    partial.replace(path)


def source_tree(archive, target, revision, offline):
    """Extract only regular files; verify every pinned source member on reuse."""
    prefix = "dinov2-" + revision
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        for member in members:
            p = Path(member.name)
            if (
                p.is_absolute()
                or ".." in p.parts
                or not p.parts
                or p.parts[0] != prefix
                or not (member.isdir() or member.isfile())
            ):
                raise ValueError("Unsafe pinned DINO source archive")
        for member in members:
            if not member.isfile():
                continue
            relative = Path(*Path(member.name).parts[1:])
            dest = target / relative
            if not dest.resolve().is_relative_to(target.resolve()):
                raise ValueError("DINO cache path escapes its source tree")
            raw = tar.extractfile(member).read()
            if dest.exists():
                if dest.is_symlink() or sha(dest) != hashlib.sha256(raw).hexdigest():
                    raise ValueError("Pinned DINO source changed: " + str(relative))
            elif offline:
                raise FileNotFoundError(dest)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(raw)


def acquire(root, lock=DEFAULT_LOCK, verify_only=False):
    pins = json.loads(lock.read_text())
    if pins["schema"] != "v2d.scored-sam3d-model-lock.v1":
        raise ValueError("Unsupported model lock")
    root = root.resolve()
    sam = pins["sam3d"]
    if not verify_only:
        from huggingface_hub import get_token, snapshot_download

        token = get_token()
        if not token:
            raise ValueError(
                "Sign in locally with hf auth login before gated downloads"
            )
        snapshot_download(
            sam["repo"],
            revision=sam["revision"],
            allow_patterns=list(sam["verified_files"]),
            local_dir=root / "hf-download",
            token=token,
            max_workers=2,
        )
    for name, record in sam["verified_files"].items():
        file = (root / "hf-download" / name).resolve()
        if not file.is_relative_to(root / "hf-download"):
            raise ValueError("Model lock path escapes download directory")
        verify(file, record)
    moge = pins["moge"]
    for record in moge["files"]:
        if verify_only:
            file = (
                root
                / "hf_home/hub/models--Ruicheng--moge-vitl/snapshots"
                / moge["revision"]
                / record["name"]
            )
        else:
            from huggingface_hub import hf_hub_download

            file = Path(
                hf_hub_download(
                    moge["repo"],
                    record["name"],
                    revision=moge["revision"],
                    cache_dir=root / "hf_home/hub",
                    token=False,
                )
            )
        verify(file, record)
    aux = pins["auxiliary"]
    hub = root / "torch_home/hub"
    archive = hub / "dinov2-source.tar.gz"
    fetch_url(
        "https://codeload.github.com/facebookresearch/dinov2/tar.gz/"
        + aux["dinov2_source_revision"],
        archive,
        {"sha256": aux["source_archive_sha256"]},
        verify_only,
    )
    source_tree(
        archive,
        hub / "facebookresearch_dinov2_main",
        aux["dinov2_source_revision"],
        verify_only,
    )
    for name, record in aux["files"].items():
        fetch_url(record["url"], hub / "checkpoints" / name, record, verify_only)
    if not verify_only:
        for name, receipt in [
            ("download-receipt.json", sam),
            ("moge-pin.json", moge),
            ("aux-receipt.json", aux),
        ]:
            (root / name).write_text(json.dumps(receipt, indent=2) + "\n")
    result = dict(
        schema="v2d.scored-sam3d-cache-verification.v1",
        model_lock_sha256=sha(lock),
        sam3d_files=len(sam["verified_files"]),
        auxiliary_weights=len(aux["files"]) + len(moge["files"]),
        source_tree_verified=True,
        credentials_in_receipt=False,
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    p.add_argument("--verify-only", action="store_true")
    a = p.parse_args()
    print(json.dumps(acquire(a.root, a.lock, a.verify_only), indent=2))
