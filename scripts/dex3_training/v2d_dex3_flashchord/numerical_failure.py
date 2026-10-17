"""Training-only numerical failures: preserve evidence, terminate and reset.

Valid transitions are unchanged. An invalid transition receives only the existing
termination penalty and a distinct numerical cause. Evaluation uses stock RLEnv
and continues to reject nonfinite worlds. No physics state is repaired here.
"""

import warp as wp
from flash_chord.envs.rl import RLEnv
from .transition_trace import TransitionTrace

ERROR_SENTINEL_M_RAD = 1000.0


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
        self.termination_index = self.objective.term_names.index("termination")
        if self.objective.weights.numpy()[self.termination_index] >= 0:
            raise ValueError(
                "Numerical failure requires the existing negative terminal penalty"
            )

    def _advance_lifecycle(self):
        self.transition_trace.record()
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
