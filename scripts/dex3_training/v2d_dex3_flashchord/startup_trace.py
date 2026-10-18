"""Passive startup/reset snapshots for a separately labelled replay diagnostic.

Reading buffers synchronizes their host copies.  It never resets, steps, clears
warm starts or writes to simulation arrays.  Missing buffers remain explicit.
This cannot by itself establish a cause of replay variability.
"""

import hashlib
import json
from pathlib import Path

import numpy as np


class StartupTrace:
    def __init__(self):
        self.arrays = {}
        self.snapshots = []

    def capture(self, stage, env, *, pending_action=None):
        """Copy selected buffers at the caller's existing lifecycle boundary."""
        index = len(self.snapshots)
        prefix = f"snapshot_{index:03d}"
        fields, missing = {}, []

        def read(path, value):
            if value is None:
                missing.append(path)
                return
            host = value.numpy() if hasattr(value, "numpy") else value
            array = np.asarray(host).copy()
            if array.dtype.hasobject:
                raise TypeError(f"Startup buffer {path} is not a numeric array")
            key = prefix + "__" + path.replace(".", "__")
            self.arrays[key] = array
            fields[path] = dict(
                key=key,
                shape=list(array.shape),
                dtype=array.dtype.str,
                sha256=hashlib.sha256(array.tobytes(order="C")).hexdigest(),
                finite=bool(np.isfinite(array).all()),
            )

        for name in ("state_0", "state_1"):
            state = getattr(env, name, None)
            for field in ("joint_q", "joint_qd", "body_q", "body_qd"):
                read(name + "." + field, getattr(state, field, None))
        solver = getattr(env, "solver", None)
        data = getattr(solver, "mjw_data", None)
        for name in ("qpos", "qvel", "qacc", "qacc_warmstart", "time", "ctrl", "qfrc_applied"):
            read("native." + name, getattr(data, name, None))
        for name in (
            "timestep", "episode_start_frame", "episode_step", "terminated",
            "truncated", "observation", "action_input", "voc_scale",
            "object_control_scale", "reset_mask",
            "sim_dt", "frame_dt", "world_count",
        ):
            read("env." + name, getattr(env, name, None))
        action = getattr(env, "action", None)
        for name in ("joint_target",):
            read("action." + name, getattr(action, name, None))
        buffers = getattr(action, "buffers", None)
        read("action.raw_action", getattr(buffers, "raw_action", None))
        control = getattr(env, "control", None)
        for name in ("joint_target_pos", "joint_target_vel", "joint_f"):
            read("control." + name, getattr(control, name, None))
        reset = getattr(env, "reset_table", None)
        for name in ("reset_frame", "all_reset_mask"):
            read("reset_table." + name, getattr(reset, name, None))
        reset_policy = getattr(env, "reset_policy", None)
        for name in ("reset_frame", "steps_since_reset", "applied_voc_scale"):
            read("reset_policy." + name, getattr(reset_policy, name, None))
        objective = getattr(env, "objective", None)
        for name in ("terms", "score"):
            read("objective." + name, getattr(objective, name, None))
        for owner_name in ("mj_model", "mjw_model"):
            option = getattr(getattr(solver, owner_name, None), "opt", None)
            for name in (
                "enableflags", "disableflags", "timestep", "solver", "integrator",
                "iterations", "ls_iterations", "cone", "impratio", "tolerance",
                "ls_tolerance", "ccd_tolerance", "ccd_iterations", "ls_parallel",
                "ls_parallel_min_step", "graph_conditional", "run_collision_detection",
                "broadphase", "broadphase_filter",
            ):
                read(owner_name + ".option." + name, getattr(option, name, None))
        if pending_action is not None:
            read("pending_policy_action", pending_action)
        self.snapshots.append(dict(index=index, stage=str(stage), fields=fields,
                                   missing_fields=missing))

    def save(self, output):
        output = Path(output)
        path = output / "startup-state-trace.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **self.arrays)
        report = dict(
            schema="v2d.passive-startup-state-trace.v1",
            snapshots=self.snapshots,
            arrays_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            simulation_buffers_written=False,
            extra_physics_steps=0,
            reset_or_warmstart_behavior_changed=False,
            official_evaluator_source_changed=False,
            buffer_readbacks_synchronize_host=True,
            replay_variability_cause_proven=False,
            scope="Buffer evidence only; no replay causality or score claim",
        )
        with (output / "startup-state-trace.json").open("x") as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        return report
