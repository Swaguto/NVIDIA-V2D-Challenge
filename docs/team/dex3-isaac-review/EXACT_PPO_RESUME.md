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

Transfer the private v8 base pack and v9 training delta directly from the owner.
Their hashes are in the [base receipt](local-pot-task-input-receipt-v8.json) and
[delta receipt](local-training-delta-receipt-v9.json). Extract both into the same
private directory, base first. The delta includes the estimated-contact inputs,
parent checkpoints, code contexts and measured outputs. Neither archive is a
GitHub download. The private input pack and GHCR access are both required.

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

## Select by recorded physical quality

Evaluate retained checkpoints with the same readback and audit commands, using
a separate fresh output folder for each. Rank those folders on the host with:

```bash
python scripts/dex3_training/select_grasp_checkpoint.py \
  --candidate /private/results/policy-grip-estimated-cws-2000-v2 \
  --candidate /private/results/policy-grip-cws-exact-resume-3500-v1 \
  --out /private/fresh-checkpoint-selection.json
```

The helper verifies each audit's report hash and unassisted replay settings.
Its worst normalized gate violation is a development diagnostic, not an official
score. Selecting the least failing checkpoint does not accept the grasp. A
passing candidate still requires an independent repeat and wider validation.

## Measured outcome

The full continuation completed 1,500 new iterations / 12,288,000 steps, ending
at global iteration 3,500 / 28,672,000 cumulative steps. It logged three failed
numerical transitions, retained in the record. The final full four-world replay
still failed: first-pot worst slip 42.1 mm, rotation 26.1°, required-pad support
97.8%, sampled depth 1.55 mm. The first full-run mean KL was 0.02075; the smoke's
was 0.02020. Complete learner restoration fixed the restart protocol, not the
physical grasp. Intermediate checkpoints also failed; the original 2,000-iteration
CWS parent remains selected. [All retained checkpoints](RETAINED_PPO_CHECKPOINTS.md).
