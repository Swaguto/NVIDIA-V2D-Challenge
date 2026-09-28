"""Leakage-safe group splitting, paired run planning, and pilot decisions."""
import random
from collections import defaultdict
from .io import identity


def make_split(records, holdout_count=4, maximum=12, seed=17):
    if not records or len({r['sequence_id'] for r in records}) != len(records):
        raise ValueError("catalog must contain unique sequences")
    if any(not r.get("permitted_for_development") or not r.get("group_id") for r in records):
        raise ValueError("every sequence needs explicit development permission and a leakage group")
    # group_id must connect shared source recordings/objects; groups never cross splits.
    groups = defaultdict(list)
    for r in records:
        for key in ("task_type", "occlusion", "hand_count", "duration_seconds"):
            if key not in r:
                raise ValueError(f"missing stratification field {key}")
        groups[r["group_id"]].append(r)
    keys = sorted(groups)
    rng = random.Random(seed)
    best = None
    for _ in range(500):
        rng.shuffle(keys)
        holdout, development = [], []
        for key in keys:
            group = groups[key]
            if len(holdout) + len(group) <= holdout_count:
                holdout.extend(group)
            elif len(development) + len(group) <= maximum - holdout_count:
                development.extend(group)
        if len(holdout) != holdout_count or len(development) < 4:
            continue
        coverage = sum(len({r[k] for r in subset}) for subset in (holdout, development)
                       for k in ("task_type", "occlusion", "hand_count"))
        candidate = (coverage, holdout, development)
        if best is None or coverage > best[0]:
            best = candidate
    if best is None:
        raise ValueError("cannot form four pilot and requested holdout sequences without splitting groups")
    _, holdout, development = best
    result = {"schema_version": "split_v1", "seed": seed, "catalog_hash": identity(records),
              "development": [r["sequence_id"] for r in development],
              "holdout": [r["sequence_id"] for r in holdout],
              "pilot": [r["sequence_id"] for r in development[:4]]}
    result["split_hash"] = identity(result)
    return result


def matrix(split, environment_steps, seeds=(17, 29, 43), phase="pilot"):
    if not isinstance(environment_steps, int) or environment_steps <= 0:
        raise ValueError("positive measured environment-step budget required")
    if phase not in ("pilot", "holdout"):
        raise ValueError("phase must be pilot or holdout")
    return [{"sequence_id": seq, "seed": seed, "variant": variant,
             "environment_steps": environment_steps, "split_hash": split["split_hash"]}
            for seq in split[phase] for seed in seeds
            for variant in ("baseline", "reliability", "constant", "shuffled")]


def pilot_decision(rows, pilot_ids, higher_is_better=True):
    import math
    paired = defaultdict(dict)
    for row in rows:
        if row["variant"] not in ("baseline", "reliability"):
            continue
        key = (row["sequence_id"], row["seed"])
        if row["sequence_id"] not in pilot_ids or row["variant"] in paired[key]:
            raise ValueError("unexpected or duplicate pilot result")
        if row["completed"] and not math.isfinite(row["metric"]):
            raise ValueError("nonfinite result")
        paired[key][row["variant"]] = row
    if set(k[0] for k in paired) != set(pilot_ids):
        raise ValueError("missing pilot sequences")
    deltas, baseline_failures, method_failures = defaultdict(list), 0, 0
    for (seq, _), pair in paired.items():
        if set(pair) != {"baseline", "reliability"}:
            raise ValueError("unpaired result")
        base, method = pair["baseline"], pair["reliability"]
        if base["environment_steps"] != method["environment_steps"]:
            raise ValueError("unequal training budgets")
        baseline_failures += not base["completed"]
        method_failures += not method["completed"]
        if base["completed"] and method["completed"]:
            deltas[seq].append((method["metric"] - base["metric"]) * (1 if higher_is_better else -1))
    wins = sum(bool(deltas[s]) and sum(deltas[s]) / len(deltas[s]) > 0 for s in pilot_ids)
    return {"advance": wins > len(pilot_ids) / 2 and method_failures <= baseline_failures,
            "sequence_wins": wins, "sequence_count": len(pilot_ids),
            "baseline_failures": baseline_failures, "method_failures": method_failures,
            "paired_deltas": dict(deltas)}
