"""Resolve Compose offline and verify the JAX cache survives container removal."""

import json
import os
import re
import subprocess
from pathlib import Path


def check(root):
    directory = Path(root) / "docker/dex3-training"
    env = os.environ.copy()
    image = "ghcr.io/example/code@sha256:" + "a" * 64
    for path in directory.glob("*.yaml"):
        for name in re.findall(r"\$\{([A-Z0-9_]+):?\?", path.read_text()):
            env[name] = (
                image
                if name.endswith("_IMAGE")
                else "/tmp/compute-config-check/" + name.lower()
            )
    names = [
        "compose.yaml",
        "compose.grip.yaml",
        "compose.repaired.yaml",
        "compose.current.yaml",
        "compose.estimated-cws.yaml",
        "compose.cws-resume.yaml",
        "compose.cws-unassisted.yaml",
        "compose.cws-retention.yaml",
        "compose.curriculum.yaml",
    ]
    for name in names:
        result = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(directory / name),
                "config",
                "--format",
                "json",
            ],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        config = json.loads(result.stdout)
        for service, item in config["services"].items():
            if (
                item["environment"].get("JAX_COMPILATION_CACHE_DIR")
                != "/root/.cache/jax"
            ):
                raise ValueError(f"Missing persistent JAX cache: {name}/{service}")
            if not any(
                v.get("target") == "/root/.cache" and v.get("type") == "volume"
                for v in item["volumes"]
            ):
                raise ValueError(
                    f"JAX cache is not in a named volume: {name}/{service}"
                )
    return {
        "compose_configs_verified": names,
        "docker_containers_started": False,
        "gpu_executed": False,
    }


if __name__ == "__main__":
    print(json.dumps(check(Path(__file__).resolve().parents[2]), indent=2))
