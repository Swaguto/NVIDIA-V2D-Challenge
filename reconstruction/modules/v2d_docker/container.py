# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import subprocess
import sys
from pathlib import Path


def _base_dir(path: str) -> str:
    """
    Return the deepest directory component of path that contains no glob characters.
    For plain paths this equals dirname(path). For paths like /out/*/*.glb it
    returns /out, so the container mount covers the full glob subtree.
    """
    parts = Path(path).parts
    clean = []
    for part in parts[:-1]:  # skip the filename/last component
        if '*' in part or '?' in part:
            break
        clean.append(part)
    return str(Path(*clean)) if clean else str(Path(path).parent)


def _extra_arg_flags(extra_args: dict[str, object] | None) -> list[str]:
    """
    Convert an extra_args mapping into CLI flags.

    None or False → omit entirely; True → bare flag (--arg_name);
    anything else → --arg_name str(value).
    """
    flags: list[str] = []
    for arg_name, value in (extra_args or {}).items():
        if value is None or value is False:
            continue
        elif value is True:
            flags.append(f"--{arg_name}")
        else:
            flags += [f"--{arg_name}", str(value)]
    return flags


def run_in_container(
    image: str,
    module: str,
    inputs: dict[str, str],
    outputs: dict[str, str],
    extra_args: dict[str, object] | None = None,
    dev: bool = False,
    modules_dir: str | None = None,
    gpus: bool = False,
    env: dict[str, str] | None = None,
    extra_volumes: list[str] | None = None,
) -> None:
    """
    Run a Python module inside a Docker container with file arguments.

    inputs:     arg_name → host_path  (read file mounts; None values are skipped)
    outputs:    arg_name → host_path  (write file mounts; parent dirs created; None skipped)
    extra_args: arg_name → value      (non-path CLI args)
                  None or False → omit entirely
                  True          → add as a bare flag (--arg_name, no value)
                  other         → add as --arg_name str(value)
    env:        extra environment variables passed via -e
    extra_volumes: raw -v arguments appended after all other mounts (e.g.
                  anonymous volumes to preserve image-built artifacts under a dev mount)

    Each unique host directory gets one volume mount at /data/<arg_name>, where the
    arg_name is taken from the first argument that references that directory. Subsequent
    arguments in the same directory reuse the same mount.
    """
    inputs  = {k: os.path.abspath(v) for k, v in inputs.items()  if v is not None}
    outputs = {k: os.path.abspath(v) for k, v in outputs.items() if v is not None}

    for path in outputs.values():
        os.makedirs(_base_dir(path), exist_ok=True)

    file_args = {**inputs, **outputs}

    # Host mode (V2D_HOST_MODE=1): run the module directly in the calling
    # Python environment, skipping Docker. Used on hosts where a container
    # runtime is unavailable (e.g. unprivileged cloud pods); requires the
    # module's lib/ dependencies to be installed on the host.
    if os.environ.get("V2D_HOST_MODE") == "1":
        cmd = [sys.executable, "-m", module]
        for arg_name, path in file_args.items():
            cmd += [f"--{arg_name}", path]
        cmd += _extra_arg_flags(extra_args)
        run_env = dict(os.environ)
        if env:
            run_env.update(env)
        subprocess.run(cmd, check=True, env=run_env)
        return

    # Map each unique host directory to a container mount point.
    # First arg to reference a directory claims the mount name.
    dir_to_mount: dict[str, str] = {}
    for arg_name, path in file_args.items():
        host_dir = _base_dir(path)
        if host_dir not in dir_to_mount:
            dir_to_mount[host_dir] = f"/data/{arg_name}"

    cmd = ["docker", "run", "--rm"]
    if gpus:
        cmd += ["--runtime=nvidia", "--gpus", "all"]
    cmd += [
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp",
    ]
    if env:
        for key, value in env.items():
            cmd += ["-e", f"{key}={value}"]
    for host_dir, container_dir in dir_to_mount.items():
        cmd += ["-v", f"{host_dir}:{container_dir}"]
    if dev:
        if modules_dir is None:
            raise ValueError("modules_dir must be provided when dev=True")
        cmd += ["-v", f"{modules_dir}:/workspace"]
    if extra_volumes:
        for vol in extra_volumes:
            cmd += ["-v", vol]

    cmd += [image, "python", "-m", module]

    for arg_name, path in file_args.items():
        host_dir = _base_dir(path)
        rel = os.path.relpath(path, host_dir)
        container_path = f"{dir_to_mount[host_dir]}/{rel}"
        cmd += [f"--{arg_name}", container_path]

    cmd += _extra_arg_flags(extra_args)

    subprocess.run(cmd, check=True)
