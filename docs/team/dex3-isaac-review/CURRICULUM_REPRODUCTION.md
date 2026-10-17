# Matched grip-training comparison

Experimental e002 development procedure, using the pinned FlashCHORD/Newton/MuJoCo-Warp backend and JAX PPO. This is not Isaac Lab/RSL-RL or a claim of successful physical grasping. e002 is outside the scored episode roster.

The GPU mechanics gate verifies identical physical models, actual 12 mm versus 35 mm wrist residual clipping, the declared VOC/reset timeline, and captured evaluation steps with zero object assistance. The code-only private image is pinned in [registry.curriculum.env.example](../../../docker/dex3-training/registry.curriculum.env.example). The image source is `035988b502637dbb7a913a96edd8ea1bcc3b56aa`; later local reward analysis is separate from its runtime.

## Inputs and host

Use a Linux NVIDIA GPU host with NVIDIA Container Toolkit and Docker Compose. The tested host has one 48 GB L40S; containers use a 24 GB host-memory limit and 8 GB shared memory. Obtain the existing [local v8 input pack](local-pot-task-input-receipt-v8.json) by private transfer, verify its hash, and extract it privately. Repository/package access does not provide the data. No parent policy is required: these runs train from scratch.

Copy `docker/dex3-training/registry.curriculum.env.example` to a private `.env.curriculum`, then set the absolute input directory to the extracted `native/e002-speed1-v8` and outputs to a fresh writable directory. Authenticate to the existing private GHCR package using your own authorized account.

```bash
docker compose --env-file .env.curriculum \
  -f docker/dex3-training/compose.curriculum.yaml pull
docker compose --env-file .env.curriculum \
  -f docker/dex3-training/compose.curriculum.yaml run --rm curriculum
```

The default command performs the GPU mechanics checks. It does not train a policy. Do not proceed if the checks fail.

## Train three matched arms

Run each command once with a fresh output path. Each arm uses seed 42, 256 worlds, 32-step rollouts, and 2,000 PPO iterations: **16,384,000 new environment steps**. Identical seeds do not guarantee bit-exact training or physics, or establish statistical robustness.

```bash
# A: current 12 mm per-axis wrist residual clip, no assistance.
docker compose --env-file .env.curriculum \
  -f docker/dex3-training/compose.curriculum.yaml run --rm curriculum \
  /opt/dex3-extension/train.py --config-dir /opt/dex3-extension/configs \
  experiment=dex3_grip_action_baseline output_dir=/outputs/narrow/ppo

# B: change only the wrist position clip to 35 mm per axis.
docker compose --env-file .env.curriculum \
  -f docker/dex3-training/compose.curriculum.yaml run --rm curriculum \
  /opt/dex3-extension/train.py --config-dir /opt/dex3-extension/configs \
  experiment=dex3_grip_reachable output_dir=/outputs/reachable/ppo

# C: same wider clip, with decaying training-only object assistance and settling.
docker compose --env-file .env.curriculum \
  -f docker/dex3-training/compose.curriculum.yaml run --rm curriculum \
  /opt/dex3-extension/train.py --config-dir /opt/dex3-extension/configs \
  experiment=dex3_grip_assisted output_dir=/outputs/assisted/ppo
```

All three arms preserve geometry, object mass, gravity, friction, native motor limits, action scale, observations, objective weights, network and optimizer settings. Finger effort remains capped at 0.76 Nm. A wider clip changes the policy's available corrections; it does not increase motor strength.

### C's training assistance

| Iterations | Target VOC | Frame-zero reset probability |
|---|---:|---:|
| 0–249 | 0.75 | 25% |
| 250–499 | 0.50 | 25% |
| 500–749 | 0.20 | 60% |
| 750–999 | 0.05 | 60% |
| 1,000–1,999 | 0.00 | 100% |

Every sampled training reset starts at VOC 1 and interpolates to the stage target over 20 control steps (one second), holding the reference until release. **The zero-target training stage still contains reset assistance.** The checkpoint records this distinction. Logger-boundary measurements record actual applied scales; they are not whole-rollout averages. This is a short development curriculum, not the paper's complete training recipe. The [CHORD implementation details](https://arxiv.org/html/2607.00033v2#A2) describe reference-state starts, settling and virtual control; unknown human contact labels remain disabled in this experiment.

## Evaluate without assistance

For each arm, replace `assisted` below with its output directory. The checkpoint evaluator forces frame-zero starts, zero VOC and virtual actuator caps, zero settling, no automatic resets, and the complete 343-frame reference. The readback verifies zero assistance before and after every control step.

```bash
docker compose --env-file .env.curriculum \
  -f docker/dex3-training/compose.curriculum.yaml run --rm curriculum \
  /opt/dex3-extension/record_policy_readback.py \
  --checkpoint /outputs/assisted/ppo/model_2000.safetensors \
  --out /outputs/assisted/readback --world-count 4
docker compose --env-file .env.curriculum \
  -f docker/dex3-training/compose.curriculum.yaml run --rm curriculum \
  /opt/dex3-extension/audit_native_task.py \
  --run /outputs/assisted/readback --inputs /data \
  --out /outputs/assisted/grasp-audit.json
```

Compare the pinned evaluator's ADD AUC, uncentered SP-SR, MP-SR and MPPE alongside **every world's** lift, slip, rotation, required-pad support, sampled penetration and source errors. A high tracking score alone does not accept the grasp. Preserve raw failures and checkpoints, then stop Brev when idle. Results require an independent repeat before any successful candidate is promoted.
