"""Verify the pinned CUDA runtime can actually execute on the host GPU."""

import argparse
import csv
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import torch

    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    devices = list(csv.reader(output.splitlines(), skipinitialspace=True))
    if len(devices) != 1:
        raise RuntimeError("This pilot requires exactly one visible GPU")
    name, driver, memory = devices[0]
    if int(driver.split(".")[0]) < 580:
        raise RuntimeError(
            f"Pinned CUDA 13 runtime requires driver 580 or newer; found {driver}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the actual simulator environment")
    value = torch.arange(32, dtype=torch.float32, device="cuda").sum()
    torch.cuda.synchronize()
    if value.item() != 496:
        raise RuntimeError("CUDA tensor calculation differs")
    report = {
        "schema": "v2d.physics-gpu-preflight.v1",
        "gpu": name,
        "driver": driver,
        "memory_total_MiB": int(memory),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "capability": list(torch.cuda.get_device_capability()),
        "cuda_tensor_sum": value.item(),
        "cuda_compute_verified": True,
        "rtx_rendering_verified": False,
    }
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
