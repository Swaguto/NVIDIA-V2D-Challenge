"""Recorded command execution, environment audit, and official evaluator adapter."""
from datetime import datetime, timezone
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from .io import read_json, write_json, sha256, identity


def capture(command):
    try:
        p = subprocess.run(command, text=True, capture_output=True, timeout=20)
        return {"returncode": p.returncode, "stdout": p.stdout.replace("\x00", "").strip(), "stderr": p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def doctor():
    return {"platform": platform.platform(), "python": sys.version,
            "disk_free_bytes": shutil.disk_usage(Path.cwd()).free,
            "gpu": capture(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv,noheader"]),
            "docker": capture(["docker", "version"]),
            "wsl": capture(["wsl", "--list", "--quiet"]) if sys.platform == "win32" else None,
            "status": "Inventory only; not proof of baseline compatibility"}


def run_recorded(command, output, config, inputs=(), artifacts=(), cwd=None):
    """Run an explicit argv without shell parsing; always record failure status."""
    if not command or not all(isinstance(a, str) for a in command):
        raise ValueError("explicit command argument list required")
    workdir = Path(cwd or Path.cwd()).resolve()
    artifacts = [Path(p) if Path(p).is_absolute() else workdir / p for p in artifacts]
    if any(p.exists() for p in artifacts):
        raise ValueError("required artifacts already exist; use fresh output paths")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    input_hashes = {str(Path(p).resolve()): sha256(p) for p in inputs}
    git = capture(["git", "-C", str(workdir), "rev-parse", "HEAD"])
    source_hashes = {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}
    manifest = {"schema_version": "run_v1", "command": command, "configuration": config,
                "started_utc": datetime.now(timezone.utc).isoformat(), "hardware": doctor(),
                "source_commit": git.get("stdout") if git.get("returncode") == 0 else None,
                "working_directory": str(workdir),
                "git_status": capture(["git", "-C", str(workdir), "status", "--porcelain"]),
                "source_checksums": source_hashes,
                "source_identity": identity(source_hashes),
                "upstream": read_json(Path(__file__).resolve().parents[2] / "configs/upstream.json")
                    if (Path(__file__).resolve().parents[2] / "configs/upstream.json").exists() else None,
                "input_checksums": input_hashes,
                "identity": identity({"config": config, "inputs": input_hashes, "command": command}),
                "peak_process_gpu_memory_bytes": None,
                "peak_memory_note": "Not measured by this portable runner; capture simulator/profiler telemetry separately.",
                "status": "running"}
    write_json(output / "manifest.json", manifest)
    started = time.monotonic()
    sampler = None
    if config.get("sample_gpu"):
        from .telemetry import DeviceSampler
        sampler = DeviceSampler()
        sampler.start()
    try:
        with (output / "stdout.log").open("w", encoding="utf-8") as stdout, (output / "stderr.log").open("w", encoding="utf-8") as stderr:
            result = subprocess.run(command, cwd=cwd, stdout=stdout, stderr=stderr, check=False)
        manifest["returncode"] = result.returncode
        manifest["status"] = "completed" if result.returncode == 0 else "failed"
        missing = [str(p) for p in artifacts if not p.is_file() or p.stat().st_size == 0]
        manifest["missing_artifacts"] = missing
        manifest["artifact_checksums"] = {str(p): sha256(p) for p in artifacts if Path(p).is_file()}
        if missing:
            manifest["status"] = "failed"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        raise
    finally:
        manifest["runtime_seconds"] = time.monotonic() - started
        if sampler is not None:
            manifest["device_telemetry"] = sampler.finish()
        write_json(output / "manifest.json", manifest)
    return manifest


def evaluate(config_path, output):
    cfg = read_json(config_path)
    if not cfg.get("rules_verified") or not cfg.get("evaluator") or not cfg.get("required_artifacts"):
        raise ValueError("official rules, evaluator and required artifacts must be supplied before evaluation")
    evaluator = Path(cfg["evaluator"]).resolve()
    if not evaluator.is_file() or sha256(evaluator) != cfg.get("evaluator_sha256"):
        raise ValueError("official evaluator missing or checksum does not match")
    if not isinstance(cfg.get("arguments"), list) or not all(isinstance(x, str) for x in cfg["arguments"]):
        raise ValueError("evaluator arguments must be a string array")
    root = Path(cfg["output_directory"]).resolve()
    artifacts = []
    for name in cfg["required_artifacts"]:
        target = (root / name).resolve()
        if not target.is_relative_to(root) or target == root:
            raise ValueError("artifact must be contained in output directory")
        if target.exists():
            raise ValueError("use a fresh evaluation directory; stale artifacts cannot prove success")
        artifacts.append(target)
    result = run_recorded([sys.executable, str(evaluator), *cfg["arguments"]], output,
                          cfg, inputs=[config_path, evaluator], artifacts=artifacts)
    result["submission_status"] = "local_artifacts_checked" if result["status"] == "completed" else "failed"
    result["official_acceptance"] = False
    write_json(Path(output) / "manifest.json", result)
    return result
