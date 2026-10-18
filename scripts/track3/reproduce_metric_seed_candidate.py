"""Reproduce the archived CPU scale/translation experiment on private source files.

This does not install a candidate in the inference pipeline or certify a grasp.
The historical producer's numerical code and file hash remain unchanged.
"""

import argparse
import ast
import hashlib
import json
import platform
from pathlib import Path

import cv2
import numpy as np
import scipy
import trimesh
from calibrated_anchor import verified_model_seed
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from select_pose_hypothesis import observed_fit

ROOT = Path(__file__).resolve().parents[2]
PRODUCER = ROOT / "experiments/scored_baseline_20261018/metric_seed_candidate_v3.py"
PRODUCER_SHA256 = "5d47bb42befc766dff7a3de385dc2f97342d1e25259b1bf405438a1109a6452d"


def archived_functions():
    if hashlib.sha256(PRODUCER.read_bytes()).hexdigest() != PRODUCER_SHA256:
        raise ValueError("Archived candidate producer changed")
    # Extract exactly the archived numerical functions. Its original machine's
    # absolute import path is historical context and is never executed here.
    tree = ast.parse(PRODUCER.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if {node.name for node in functions} != {"sha", "run"}:
        raise ValueError("Unexpected archived producer functions")
    namespace = dict(
        __file__=str(PRODUCER),
        Path=Path,
        hashlib=hashlib,
        json=json,
        cv2=cv2,
        np=np,
        trimesh=trimesh,
        minimize=minimize,
        cKDTree=cKDTree,
        observed_fit=observed_fit,
    )
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(PRODUCER), "exec"),
        namespace,
    )
    return namespace


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--support-report", type=Path)
    p.add_argument("--bundle", type=Path)
    p.add_argument("--initial-candidate", type=Path)
    p.add_argument(
        "--seed-known-over-table",
        action="store_true",
        help="Required for the historical pilot plane constraint; no off-table inference is supplied",
    )
    a = p.parse_args()
    if (a.support_report is None) != (a.bundle is None):
        p.error("Support report and original camera bundle must be supplied together")
    if a.support_report is not None and not a.seed_known_over_table:
        p.error(
            "This archived plane experiment requires an independently known over-table seed"
        )
    proof = json.loads((a.source / "manifest.json").read_text())
    verified_model_seed(a.source / "manifest.json", proof)
    functions = archived_functions()
    functions["run"](a.source, a.out, a.support_report, a.bundle, a.initial_candidate)
    environment = dict(
        schema="v2d.archived-metric-seed-reproduction-environment.v1",
        producer_sha256=PRODUCER_SHA256,
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        observed_fit_source_sha256=hashlib.sha256(
            (ROOT / "scripts/track3/select_pose_hypothesis.py").read_bytes()
        ).hexdigest(),
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        opencv=cv2.__version__,
        trimesh=trimesh.__version__,
        candidate_installed_in_pipeline=False,
        physical_task_accepted=False,
    )
    (a.out / "reproduction-environment.json").write_text(
        json.dumps(environment, indent=2) + "\n"
    )
