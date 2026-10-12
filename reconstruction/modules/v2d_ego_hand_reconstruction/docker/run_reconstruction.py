# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run ego hand reconstruction pipeline (ViPE camera estimation + Dyn-HaMR).

Delegates to the vendored ``run_reconstruction.sh`` so pipeline logic stays in
sync with upstream IsaacTeleop.

Prerequisites:
    - Both Docker images must be built (``python -m v2d_ego_hand_reconstruction.docker.build``).
    - ``weights_dir`` must contain ``models/MANO_RIGHT.pkl`` and ``BMC/*.npy``
      (manotorch layout; see https://nvidia.github.io/IsaacTeleop/main/references/egocentric_hand_reconstruction.html for download instructions).
"""

import os
import shutil
import subprocess

from v2d_ego_hand_reconstruction.docker._config import VENDOR_DIR

_HOST = "/home/appuser"


def _host_link(target: str, link: str) -> None:
    """Create ``link -> target`` (mimics a container volume mount).

    An existing empty directory is replaced (some tools create it eagerly);
    a non-empty directory or a link pointing elsewhere is an error.
    """
    if os.path.islink(link):
        if os.path.realpath(link) == os.path.realpath(target):
            return
        raise RuntimeError(f"{link} already links to {os.path.realpath(link)}")
    if os.path.isdir(link):
        if os.listdir(link):
            raise RuntimeError(f"{link} is a non-empty directory; cannot bind to {target}")
        os.rmdir(link)
    elif os.path.exists(link):
        raise RuntimeError(f"{link} exists and is not a directory")
    os.makedirs(os.path.dirname(link), exist_ok=True)
    os.symlink(os.path.abspath(target), link)


def _run_reconstruction_host(video_input: str, output_dir: str) -> None:
    """V2D_HOST_MODE=1: run the vendored pipeline steps directly on the host.

    Reproduces the container layout from ``vendor/docker/vipe.sh`` and
    ``dynhamr.sh`` as symlinks under /home/appuser, then runs the same
    commands those wrappers issue, via the conda environments built by the
    bootstrap script (``bootstrap_v2d_host.sh vipe|dynhamr``).
    """
    if video_input.startswith(("s3://", "swift://")):
        raise RuntimeError("V2D_HOST_MODE does not support bucket URLs")

    video_file = os.path.basename(video_input)
    if not os.path.isfile(video_input):
        raise RuntimeError(f"video not found: {video_input}")
    staged = os.path.join(output_dir, video_file)
    if not os.path.exists(staged):
        shutil.copy2(video_input, staged)

    conda_bin = os.path.join(_HOST, "miniconda3", "bin")
    conda = os.path.join(conda_bin, "conda")
    for required in (
        conda,
        os.path.join(_HOST, "ViPE"),
        os.path.join(_HOST, "Dyn-HaMR", "dyn-hamr"),
        os.path.join(_HOST, "setup_dynhamr.sh"),
    ):
        if not os.path.exists(required):
            raise RuntimeError(f"missing {required}; run bootstrap stages vipe/dynhamr first")

    # Container mounts -> host symlinks (same absolute paths as in-container).
    _host_link(output_dir, os.path.join(_HOST, "outputs"))
    _host_link(output_dir, os.path.join(_HOST, "ViPE", "vipe_results"))
    _host_link(output_dir, os.path.join(_HOST, "Dyn-HaMR", "outputs"))

    env = {
        **os.environ,
        "OUTPUTS_DIR": output_dir,
        "HOME": _HOST,
        "TORCH_HOME": os.path.join(_HOST, ".cache", "torch"),
        "HF_HOME": os.path.join(_HOST, ".cache", "huggingface"),
        "PATH": f"{conda_bin}:{os.environ.get('PATH', '')}",
    }

    # Step 2 — vendor/scripts/run_reconstruction.sh:105
    subprocess.run(
        [conda, "run", "--no-capture-output", "-n", "vipe",
         "vipe", "infer", f"vipe_results/{video_file}"],
        cwd=os.path.join(_HOST, "ViPE"),
        env=env,
        check=True,
    )

    # Step 3 — vendor/scripts/run_reconstruction.sh:111-127
    subprocess.run(
        [conda, "run", "--no-capture-output", "-n", "dynhamr",
         "bash", os.path.join(_HOST, "setup_dynhamr.sh")],
        env=env,
        check=True,
    )
    subprocess.run(
        ["ln", "-sf",
         os.path.join(output_dir, video_file),
         os.path.join(_HOST, "Dyn-HaMR", "test", "videos", video_file)],
        env=env,
        check=True,
    )
    video_name = os.path.splitext(video_file)[0]
    subprocess.run(
        [conda, "run", "--no-capture-output", "-n", "dynhamr", "bash", "-c",
         "cd /home/appuser/Dyn-HaMR/dyn-hamr && python run_opt.py "
         "data=video_vipe run_opt=True run_vis=True "
         f"data.seq={video_name} is_static=False"],
        cwd=os.path.join(_HOST, "Dyn-HaMR", "dyn-hamr"),
        env=env,
        check=True,
    )


def run_reconstruction(
    video_input: str,
    output_dir: str,
    weights_dir: str,
) -> None:
    """Run the full ego hand reconstruction pipeline.

    Args:
        video_input: Local file path, or ``s3://`` / ``swift://`` URL.
        output_dir: Directory for all pipeline outputs.
        weights_dir: Directory containing ``models/MANO_RIGHT.pkl`` and ``BMC/``
            (manotorch layout — shared with v2d_hamer).
    """
    output_dir = os.path.abspath(output_dir)
    weights_dir = os.path.abspath(weights_dir)
    os.makedirs(output_dir, exist_ok=True)

    # The vendored Dyn-HaMR container reads MANO_RIGHT.pkl from the top of
    # output_dir, so copy it out of the manotorch-style models/ subdir.
    # Symlinking doesn't work because the target is outside the container's mount.
    mano_src = os.path.join(weights_dir, "models", "MANO_RIGHT.pkl")
    mano_dst = os.path.join(output_dir, "MANO_RIGHT.pkl")
    if not os.path.exists(mano_dst):
        shutil.copy2(mano_src, mano_dst)

    bmc_src = os.path.join(weights_dir, "BMC")
    bmc_dst = os.path.join(output_dir, "BMC")
    if not os.path.exists(bmc_dst):
        shutil.copytree(bmc_src, bmc_dst)

    # Resolve local paths to absolute so the vendored script can find them
    if not video_input.startswith(("s3://", "swift://")):
        video_input = os.path.abspath(video_input)

    if os.environ.get("V2D_HOST_MODE") == "1":
        _run_reconstruction_host(video_input, output_dir)
        return

    subprocess.run(
        [os.path.join(VENDOR_DIR, "scripts", "run_reconstruction.sh"), video_input],
        env={**os.environ, "OUTPUTS_DIR": output_dir},
        check=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video_input",
        required=True,
        help="Local path or s3:///swift:// URL",
    )
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--weights_dir", required=True, help="Directory with models/MANO_RIGHT.pkl and BMC/")
    args = parser.parse_args()
    run_reconstruction(args.video_input, args.output_dir, args.weights_dir)
