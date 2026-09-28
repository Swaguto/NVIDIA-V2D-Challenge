import sys
import pytest
from v2d_reliability.experiments import make_split, matrix, pilot_decision
from v2d_reliability.io import write_json, sha256
from v2d_reliability.workflow import run_recorded, evaluate


def catalog():
    return [{"sequence_id": f"s{i}", "group_id": f"g{i // 2}", "permitted_for_development": True,
             "task_type": str(i % 3), "occlusion": str(i % 2), "hand_count": 2, "duration_seconds": 10}
            for i in range(12)]


def test_groups_and_matrix():
    records = catalog(); split = make_split(records)
    assert split == make_split(records)
    groups = {r["sequence_id"]: r["group_id"] for r in records}
    assert not {groups[s] for s in split["development"]} & {groups[s] for s in split["holdout"]}
    jobs = matrix(split, 1000)
    assert len(jobs) == 4 * 3 * 4
    assert {j["environment_steps"] for j in jobs} == {1000}
    records[0]["permitted_for_development"] = False
    with pytest.raises(ValueError): make_split(records)


def test_pilot_gate_requires_pairs_and_majority():
    ids = [f"s{i}" for i in range(4)]
    rows = [{"sequence_id": s, "seed": 17, "variant": v, "environment_steps": 100,
             "completed": True, "metric": .5 if v == "baseline" else .6} for s in ids for v in ["baseline", "reliability"]]
    assert pilot_decision(rows, ids)["advance"]
    with pytest.raises(ValueError): pilot_decision(rows[:-1], ids)
    rows[-1]["completed"] = False
    assert not pilot_decision(rows, ids)["advance"]


def test_runner_records_failures(tmp_path):
    result = run_recorded([sys.executable, "-c", "raise SystemExit(3)"], tmp_path / "run", {})
    assert result["status"] == "failed" and result["returncode"] == 3
    assert (tmp_path / "run/manifest.json").is_file()


def test_artifact_paths_use_working_directory_and_ledger_keeps_failure(tmp_path):
    from v2d_reliability.ledger import collect_runs
    work = tmp_path / "work"; work.mkdir()
    result = run_recorded([sys.executable, "-c", "from pathlib import Path; Path('result.txt').write_text('fixture')"],
                          tmp_path / "runs/ok", {"variant": "baseline"}, artifacts=["result.txt"], cwd=work)
    assert result["status"] == "completed"
    run_recorded([sys.executable, "-c", "raise SystemExit(7)"], tmp_path / "runs/bad", {})
    ledger = collect_runs(tmp_path / "runs")
    assert {r["status"] for r in ledger["runs"]} == {"completed", "failed"}
    assert all(not r["training_success_verified"] for r in ledger["runs"])
    with pytest.raises(ValueError):
        run_recorded([sys.executable, "-c", "pass"], tmp_path / "runs/stale", {}, artifacts=["result.txt"], cwd=work)


def test_official_evaluator_gate_and_missing_artifact(tmp_path):
    config = tmp_path / "config.json"
    write_json(config, {"rules_verified": False})
    with pytest.raises(ValueError): evaluate(config, tmp_path / "not_created")
    assert not (tmp_path / "not_created").exists()
    evaluator = tmp_path / "eval.py"; evaluator.write_text("print('fixture evaluator, not official')")
    cfg = {"rules_verified": True, "evaluator": str(evaluator), "evaluator_sha256": sha256(evaluator),
           "arguments": [], "output_directory": str(tmp_path / "artifacts"), "required_artifacts": ["result.zip"]}
    write_json(config, cfg)
    result = evaluate(config, tmp_path / "run")
    assert result["status"] == "failed" and not result["official_acceptance"]
    cfg["required_artifacts"] = ["../escape.zip"]; write_json(config, cfg)
    with pytest.raises(ValueError): evaluate(config, tmp_path / "escape")
