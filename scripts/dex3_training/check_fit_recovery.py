"""Inject one native failure into a real search, then require a clean full replay."""

import json
from pathlib import Path

import numpy as np
import warp as wp
import fit_phase_policy as fitting


@wp.kernel
def inject_invalid_cache(world: int, bad: wp.array(dtype=wp.float32),
                         cache: wp.array2d(dtype=wp.float32)):
    dof = wp.tid()
    cache[world, dof] = bad[0]


class InjectedSearchEnv(fitting.RLEnv):
    instance = None

    def __init__(self, *args, **kwargs):
        self.contract_steps = 0
        self.contract_injected = False
        super().__init__(*args, **kwargs)
        type(self).instance = self

    def step(self, action=None):
        self.contract_steps += 1
        if self.contract_steps == 60:
            data = self.solver.mjw_data
            wp.launch(inject_invalid_cache, dim=data.qacc_warmstart.shape[1],
                      inputs=[self.world_count - 1,
                              wp.array([np.nan], dtype=wp.float32, device=self.device),
                              data.qacc_warmstart], device=self.device)
            self.contract_injected = True
        return super().step(action)


if __name__ == "__main__":
    fitting.RLEnv = InjectedSearchEnv
    fitting.main()
    env = InjectedSearchEnv.instance
    assert env.contract_injected and env.world_count == 8
    out = Path("/outputs/fit")
    rows = [json.loads(line) for line in (out / "iterations.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["discarded_proposal_indices"] == [3]
    assert row["native_motor_caps_verified"] and row["scored_cohort_all_finite"]
    assert row["reset_native_warmstart_cleared"]
    assert row["simulated_environment_steps"] == 2 * 8 * 343
    assert row["discarded_cohort_environment_steps"] == 8 * 343
    assert np.isfinite(env.state_0.body_q.numpy()).all()
    with np.load(out / "failed-cohort-1.npz") as raw:
        assert not np.isfinite(raw["body_poses"][:, 7]).all()
    result = dict(schema="v2d.native-fit-recovery-contract.v1",
                  injected_native_failure_observed=True,
                  failed_batch_scored=False, discarded_candidate_pair=3,
                  complete_clean_replay_frames=343, clean_worlds=8,
                  native_caps_verified=True, failed_states_repaired=False,
                  physical_task_accepted=False)
    (out / "recovery-contract.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
