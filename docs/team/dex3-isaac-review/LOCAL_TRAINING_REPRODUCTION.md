# Local e002 training reproduction bundle

`e002-training-reproduction-local-v1.tar.gz` is a private local artifact. It has
not been uploaded to GitHub or included in a container. Obtain it from the owner
through an approved private channel. `local-training-bundle-receipt-v1.json`
contains its size and SHA-256; verify those before extracting into a fresh folder.
The archive includes the frozen input pack, five checkpoints including the initial
policy, metrics, physics readback, original evaluation results and a per-file hash
manifest. It contains no MANO weights or credentials.

Use the published image on Linux amd64 with the GPU/driver requirements in
`docker/dex3-training/README.md`. From this repository checkout:

```bash
set -a
. docker/dex3-training/registry.env.example
set +a
export DEX3_TRAINING_INPUTS=/private/extracted-bundle/inputs
export DEX3_TRAINING_OUTPUTS=/private/extracted-bundle/outputs
docker pull "$DEX3_TRAINING_IMAGE"

docker compose -f docker/dex3-training/compose.yaml run --rm pilot \
  /opt/dex3-extension/runtime_controls.py \
  --reference /data/reference.parquet --out /outputs/runtime-controls-repeat.json

docker compose -f docker/dex3-training/compose.yaml run --rm pilot \
  /opt/flash-chord/scripts/evaluate_policy.py \
  evaluation.checkpoint=/outputs/exploration-locked-v1/model_0.safetensors \
  evaluation.world_count=4 \
  evaluation.metrics_output=/outputs/eval-repeat-initial.json \
  evaluation.object_trajectories_output=/outputs/eval-repeat-initial.parquet

docker compose -f docker/dex3-training/compose.yaml run --rm pilot \
  /opt/flash-chord/scripts/evaluate_policy.py \
  evaluation.checkpoint=/outputs/exploration-locked-v1/model_1000.safetensors \
  evaluation.world_count=4 \
  evaluation.metrics_output=/outputs/eval-repeat-1000.json \
  evaluation.object_trajectories_output=/outputs/eval-repeat-1000.parquet

python3.11 scripts/dex3_training/compare_policies.py \
  --inputs "$DEX3_TRAINING_INPUTS" \
  --initial-checkpoint "$DEX3_TRAINING_OUTPUTS/exploration-locked-v1/model_0.safetensors" \
  --learned-checkpoint "$DEX3_TRAINING_OUTPUTS/exploration-locked-v1/model_1000.safetensors" \
  --initial-report "$DEX3_TRAINING_OUTPUTS/eval-repeat-initial.json" \
  --learned-report "$DEX3_TRAINING_OUTPUTS/eval-repeat-1000.json" \
  --runtime-controls "$DEX3_TRAINING_OUTPUTS/runtime-controls-repeat.json" \
  --out "$DEX3_TRAINING_OUTPUTS/comparison-repeat.json"
```

The comparator requires only Python's standard library. Fresh report names avoid
overwriting the archived evidence. The measured development outcome is failure
for both policies; this bundle reproduces an experiment, not a solved task.
Small floating-point/simulation differences can remain. Bit-exact retraining and
hidden-set success are not claimed.

To retrain from scratch, set `DEX3_TRAINING_OUTPUTS` to a new writable directory
and run the `dex3_exploration` command in the Docker README. Keep the frozen input
directory read-only. Stop the GPU VM after backing up results and whenever idle.
