"""Fork only the pinned learner's PRNG state for paired continuation repeats.

This is explicitly not an unchanged RNG resume or a new actor initialization.
The pinned LearnerState has one legacy uint32[2] key. Ambiguous envelopes fail.
Actor, critic, optimizer, normalization, observation and counters remain exact.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rekey_arrays(arrays, manifest, key):
    if {r["key"] for r in manifest} != set(arrays) or len(manifest) != len(arrays):
        raise ValueError("Learner manifest and tensors differ")
    key = np.asarray(key)
    if key.shape != (2,) or key.dtype != np.uint32:
        raise ValueError("Pinned learner requires a legacy uint32[2] PRNG key")
    candidates = [
        r["key"]
        for r in manifest
        if tuple(r["shape"]) == (2,) and np.dtype(r["dtype"]) == np.dtype("uint32")
    ]
    if len(candidates) != 1:
        raise ValueError("Ambiguous or missing learner PRNG leaf")
    name = candidates[0]
    if arrays[name].shape != (2,) or arrays[name].dtype != np.uint32:
        raise ValueError("PRNG array and envelope differ")
    result = dict(arrays)
    result[name] = key.copy()
    if np.array_equal(result[name], arrays[name]):
        raise ValueError("New key must differ from the parent")
    return result, name


def fork(parent, expected, seed, out):
    import jax
    from flash_chord.training.ppo import learner
    from safetensors import safe_open
    from safetensors.numpy import load_file, save_file

    if (
        sha(Path(learner.__file__))
        != "001fea28cdf2a5fbe56891a3627d9e4fdc9a30ffc228a32be1b42eff91c2753e"
    ):
        raise ValueError("PRNG fork requires the pinned upstream LearnerState layout")

    if sha(parent) != expected:
        raise ValueError("Parent checkpoint SHA256 differs")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Use an explicit unsigned 32-bit exploration seed")
    if out.exists():
        raise FileExistsError(out)
    with safe_open(str(parent), framework="numpy") as f:
        metadata = f.metadata()
    if metadata.get("format") != "learner_state" or metadata.get("version") != "1":
        raise ValueError("Unknown pinned learner envelope")
    if "user.dex3_rng_fork_json" in metadata:
        raise ValueError("Start each repeat from the unchanged original checkpoint")
    arrays = load_file(str(parent))
    result, name = rekey_arrays(
        arrays, json.loads(metadata["manifest"]), np.asarray(jax.random.PRNGKey(seed))
    )
    report = {
        "schema": "v2d.paired-continuation-prng-fork.v1",
        "original_checkpoint_sha256": expected,
        "exploration_seed": seed,
        "changed_tensor": name,
        "learner_source_sha256": sha(Path(learner.__file__)),
        "parent_key": arrays[name].tolist(),
        "key": result[name].tolist(),
        "actor_critic_optimizer_normalization_exact": True,
        "training_counters_changed": False,
        "simulator_state_restored": False,
        "unchanged_rng_resume_claimed": False,
        "environment_reset_seed_changed": False,
        "producer_sha256": sha(Path(__file__)),
    }
    metadata["user.dex3_rng_fork_json"] = json.dumps(report, sort_keys=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(result, str(out), metadata=metadata)
    restored = load_file(str(out))
    for tensor, value in arrays.items():
        expected_value = result[tensor] if tensor == name else value
        if (
            restored[tensor].dtype != expected_value.dtype
            or restored[tensor].shape != expected_value.shape
            or restored[tensor].tobytes() != expected_value.tobytes()
        ):
            raise ValueError("Fork changed an undeclared learner tensor")
    report["fork_checkpoint_sha256"] = sha(out)
    report["unchanged_tensors_verified"] = len(arrays) - 1
    out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--expected-parent-sha256", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(fork(a.parent, a.expected_parent_sha256, a.seed, a.out), indent=2))
