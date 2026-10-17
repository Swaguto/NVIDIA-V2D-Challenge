"""Training-only numerical failures: preserve evidence, terminate and reset.

Valid transitions are unchanged. An invalid transition receives only the existing
termination penalty and a distinct numerical cause. Evaluation uses stock RLEnv
and continues to reject nonfinite worlds. Failed transitions are never repaired;
only reset worlds discard the previous episode's native solver warm start.
"""

import warp as wp
from flash_chord.envs.rl import RLEnv
from .transition_trace import TransitionTrace

ERROR_SENTINEL_M_RAD = 1000.0


@wp.kernel
def clear_reset_warmstart(
    selected: wp.array(dtype=wp.int32),
    warmstart: wp.array2d(dtype=wp.float32),
):
    world, dof = wp.tid()
    if selected[world] != 0:
        warmstart[world, dof] = 0.0


@wp.kernel
def track_failure_streak(
    flags: wp.array(dtype=wp.int32),
    streak: wp.array(dtype=wp.int32),
    maximum: wp.array(dtype=wp.int32),
):
    world = wp.tid()
    if flags[world] != 0:
        streak[world] = streak[world] + 1
        maximum[world] = wp.max(maximum[world], streak[world])
    else:
        streak[world] = 0


@wp.kernel
def publish_diagnostics(
    base: wp.array(dtype=wp.float32),
    numerical: wp.array(dtype=wp.int32),
    errors0: wp.array(dtype=wp.float32),
    errors1: wp.array(dtype=wp.float32),
    errors2: wp.array(dtype=wp.float32),
    errors3: wp.array(dtype=wp.float32),
    output: wp.array(dtype=wp.float32),
):
    world = wp.tid()
    numerical[world] = 0
    output[world * 7] = base[world * 6]
    output[world * 7 + 1] = base[world * 6 + 1]
    output[world * 7 + 2] = 0.0
    for axis in range(4):
        output[world * 7 + 3 + axis] = base[world * 6 + 2 + axis]
    errors0[world] = base[world * 6 + 2]
    errors1[world] = base[world * 6 + 3]
    errors2[world] = base[world * 6 + 4]
    errors3[world] = base[world * 6 + 5]


@wp.kernel
def terminate_invalid_transition(
    flags: wp.array(dtype=wp.int32),
    terms: int,
    termination_index: int,
    dt: float,
    weights: wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.int32),
    truncated: wp.array(dtype=wp.int32),
    wrist: wp.array(dtype=wp.int32),
    objects: wp.array(dtype=wp.int32),
    numerical: wp.array(dtype=wp.int32),
    objective: wp.array(dtype=wp.float32),
    score: wp.array(dtype=wp.float32),
    diagnostics: wp.array(dtype=wp.float32),
    errors0: wp.array(dtype=wp.float32),
    errors1: wp.array(dtype=wp.float32),
    errors2: wp.array(dtype=wp.float32),
    errors3: wp.array(dtype=wp.float32),
):
    world = wp.tid()
    if flags[world] == 0:
        return
    terminated[world] = 1
    truncated[world] = 0
    wrist[world] = 0
    objects[world] = 0
    numerical[world] = 1
    for term in range(terms):
        objective[world * terms + term] = 0.0
    objective[world * terms + termination_index] = 1.0
    score[world] = dt * weights[termination_index]
    diagnostics[world * 7] = 0.0
    diagnostics[world * 7 + 1] = 0.0
    diagnostics[world * 7 + 2] = 1.0
    for axis in range(4):
        diagnostics[world * 7 + 3 + axis] = 1000.0
    errors0[world] = 1000.0
    errors1[world] = 1000.0
    errors2[world] = 1000.0
    errors3[world] = 1000.0


class TrainingTermination:
    def __init__(self, base, worlds, device):
        if base.cause_names != ("wrist", "object") or len(base.error_names) != 4:
            raise ValueError(
                "Numerical termination supports the floating-hand diagnostics"
            )
        self.base = base
        self.worlds = worlds
        self.terminated, self.truncated = base.terminated, base.truncated
        self.numerical = wp.zeros(worlds, dtype=wp.int32, device=device)
        self.errors = tuple(
            wp.zeros(worlds, dtype=wp.float32, device=device) for _ in range(4)
        )
        self.packed_diagnostics = wp.zeros(worlds * 7, dtype=wp.float32, device=device)

    @property
    def cause_names(self):
        return self.base.cause_names + ("numerical",)

    @property
    def cause_arrays(self):
        return self.base.cause_arrays + (self.numerical,)

    @property
    def error_names(self):
        return self.base.error_names

    @property
    def error_arrays(self):
        return self.errors

    @property
    def packed_diagnostic_names(self):
        return self.cause_names + self.error_names

    def evaluate(self, body_q, timestep):
        self.base.evaluate(body_q, timestep)
        wp.launch(
            publish_diagnostics,
            dim=self.worlds,
            inputs=[
                self.base.packed_diagnostics,
                self.numerical,
                *self.errors,
                self.packed_diagnostics,
            ],
        )


class NumericalFailureRLEnv(RLEnv):
    numerical_failure_policy = "terminate"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.config.auto_reset or self.config.capture_terminal_state:
            raise ValueError(
                "Numerical termination is training-only with automatic resets"
            )
        self.termination = TrainingTermination(
            self.termination, self.world_count, self.device
        )
        self.transition_trace = TransitionTrace(self)
        self.failure_streak = wp.zeros(
            self.world_count, dtype=wp.int32, device=self.device
        )
        self.maximum_failure_streak = wp.zeros_like(self.failure_streak)
        data = self.solver.mjw_data
        if data is None or data.nworld != self.world_count:
            raise ValueError("Numerical resets require the pinned MuJoCo-Warp backend")
        if self.solver.mjw_model.na != 0:
            raise ValueError("Stateful actuators need a separate reset contract")
        self.termination_index = self.objective.term_names.index("termination")
        if self.objective.weights.numpy()[self.termination_index] >= 0:
            raise ValueError(
                "Numerical failure requires the existing negative terminal penalty"
            )

    def _clear_reset_warmstart(self, selected):
        # MuJoCo-Warp 3.8.1 _solve reads qacc_warmstart before solving. Newton
        # updates qpos/qvel from reset state each step, but does not clear this
        # cache. Do not call reset_data: it also modifies globally packed contacts.
        warmstart = self.solver.mjw_data.qacc_warmstart
        wp.launch(
            clear_reset_warmstart,
            dim=warmstart.shape,
            inputs=[selected, warmstart],
        )

    def reset(self, frame_id=None, reset_frame=None, reset_mask=None):
        observation = super().reset(
            frame_id=frame_id, reset_frame=reset_frame, reset_mask=reset_mask
        )
        selected = reset_mask
        if selected is None:
            selected = self.reset_table.all_reset_mask
        self._clear_reset_warmstart(selected)
        return observation

    def _rl_step(self):
        super()._rl_step()
        # BaseEnv.reset is called directly inside upstream _rl_step, so this
        # must run here as well as in the explicit reset method. CUDA captured.
        self._clear_reset_warmstart(self.reset_mask)

    def _advance_lifecycle(self):
        self.transition_trace.record()
        wp.launch(
            track_failure_streak,
            dim=self.world_count,
            inputs=[
                self.transition_trace.latest_flags,
                self.failure_streak,
                self.maximum_failure_streak,
            ],
        )
        t = self.termination
        wp.launch(
            terminate_invalid_transition,
            dim=self.world_count,
            inputs=[
                self.transition_trace.latest_flags,
                len(self.objective.term_names),
                self.termination_index,
                self.frame_dt,
                self.objective.weights,
                self.terminated,
                self.truncated,
                *t.base.cause_arrays,
                t.numerical,
                self.objective.terms,
                self.objective.score,
                t.packed_diagnostics,
                *t.errors,
            ],
        )
        super()._advance_lifecycle()
