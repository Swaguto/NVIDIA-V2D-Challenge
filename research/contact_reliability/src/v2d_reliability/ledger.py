"""Rebuild an experiment ledger from immutable run manifests, including failures."""
from pathlib import Path
from .io import read_json, sha256


def collect_runs(root):
    rows = []
    for path in sorted(Path(root).rglob("manifest.json")):
        run = read_json(path)
        if run.get("schema_version") != "run_v1":
            continue
        cfg = run["configuration"]
        rows.append({"manifest": str(path.resolve()), "manifest_sha256": sha256(path),
                     "hypothesis": cfg.get("hypothesis"), "episode_id": cfg.get("episode_id"),
                     "variant": cfg.get("variant"), "seed": cfg.get("seed"),
                     "source_commit": run.get("source_commit"), "inputs": run["input_checksums"],
                     "configuration": cfg, "status": run["status"],
                     "runtime_seconds": run.get("runtime_seconds"),
                     "failure_reason": run.get("error") or run.get("missing_artifacts") or
                         (f"exit {run.get('returncode')}" if run["status"] == "failed" else None),
                     "evidence_level": "command_execution",
                     "training_success_verified": False,
                     "official_acceptance": run.get("official_acceptance", False)})
    return {"schema_version": "ledger_v1", "runs": rows,
            "note": "Successful commands do not prove policy learning or official acceptance."}
