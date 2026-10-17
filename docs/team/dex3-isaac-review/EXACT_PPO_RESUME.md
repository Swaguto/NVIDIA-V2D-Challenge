# Continue the complete CWS learner

The CWS parent reached 2,000 iterations / 16,384,000 steps. Its adaptive learning
rate ended at `1e-5`, with learned exploration standard deviations about
0.099–0.120. The actor-only experiments started a fresh critic and optimizer at
`3e-4` and exploration standard deviation 0.15. Their first PPO iterations
reported mean KL about 2.1 against a configured target of 0.005. The adaptive
learning rate had already fallen to `1e-5` by the end of that first iteration.
They also changed the
training reset regime; these diagnostics do not isolate a single cause of their
regression.

The exact continuation restores actor, critic, learned exploration, optimizer
moments, RNG, normalization, adaptive learning rate and counters. It retains the
original training recipe, including training reset stabilization. Independent
evaluation disables all object assistance and stabilization.

The GPU restoration check exercises the real pinned runner with zero updates.
All learner tensors must equal the loaded parent, while observations must come
from the fresh environment reset. Its smoke then performs eight actual new PPO
iterations. The observed first resumed KL was 0.0202, versus 2.0969 in the
actor-only control. This is an optimization diagnostic; physical quality still
requires the complete replay and grasp gates.

## Reproduction

Use the parent checkpoint's original immutable image, supplied in
[registry.cws-resume.env.example](../../../docker/dex3-training/registry.cws-resume.env.example).
Set private input, parent-policy, parent-context and fresh output paths, plus the
absolute path to the checked-in `check_exact_resume.py` file. The companion
checker is mounted read-only; the original runtime image remains unchanged.

```bash
docker compose --env-file /path/to/local-resume.env \
  -f docker/dex3-training/compose.cws-resume.yaml run --rm cws-resume \
  /checks/check_exact_resume.py \
  --checkpoint /initial-policy/ppo/model_2000.safetensors \
  --out /outputs/resume-state-proof.json
docker compose --env-file /path/to/local-resume.env \
  -f docker/dex3-training/compose.cws-resume.yaml run --rm cws-resume
```

The default smoke ends at **global iteration 2008**, adding 65,536 steps. For the
full continuation, use a fresh output directory and set
`DEX3_RESUME_END_ITERATION=3500` and `DEX3_RESUME_SAVE_INTERVAL=100` in a separate
local environment file. Keep `DEX3_RESUME_PARENT_ITERATION=2000`. That full run
adds **1,500 iterations / 12,288,000 steps**, ending at 28,672,000 cumulative steps.
The smoke is a separate check and does not initialize the full continuation.

After training, record and audit the entire unassisted episode:

```bash
docker compose --env-file /path/to/local-full-resume.env \
  -f docker/dex3-training/compose.cws-resume.yaml run --rm cws-resume \
  /opt/dex3-extension/record_policy_readback.py \
  --checkpoint /outputs/ppo/model_3500.safetensors --out /outputs/diagnostic \
  --world-count 4
docker compose --env-file /path/to/local-full-resume.env \
  -f docker/dex3-training/compose.cws-resume.yaml run --rm cws-resume \
  /opt/dex3-extension/audit_native_task.py \
  --run /outputs/diagnostic --inputs /data --out /outputs/audit.json
```

The simulator resets on resume; its trajectory is not promised bit-exact.
Inputs and weights stay private. e002 is a development episode outside the scored
roster; no hidden-episode result follows from this test.

All four inherited Compose services were resolved by Docker Compose on the Brev
host. That check verified GPU reservation, read-only input and parent mounts,
writable output mounts, a 24 GiB memory limit and 8 GiB shared memory. It did not
start another training container.
