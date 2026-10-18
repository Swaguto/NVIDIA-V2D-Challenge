# Repeat a frozen scored policy

This repeats an archived policy's official evaluation and full native readback.
It adds no training steps. It verifies frozen file hashes, the checkpoint's input
binding, the producer's immutable image and the complete eight-world source
clock before Docker. Source, checkpoint and kit mounts are read-only; containers
have no network access and refuse implicit image pulls.

The owner must privately transfer the matching frozen inputs, checkpoint,
producer receipt and saved official evaluation report. These files are not
published. The teammate also needs package read access, the unchanged official
kit, Linux amd64, Docker with NVIDIA Container Toolkit and a compatible GPU.
Keep inference weights and footage separate: they are not needed for this
policy-only replay. Full reconstruction reproduction still needs the private
model and video prerequisites in the scored Docker guide.

```sh
python scripts/track3/replay_scored_episode.py \
  --inputs /private/frozen \
  --checkpoint /private/model_100.safetensors \
  --expected-report /private/episode_000015.json \
  --policy-receipt /private/producer-receipt.json \
  --kit /private/v2d_submission_kit \
  --out /private/fresh-replay
```

The default only prints a verified plan. It neither invokes Docker nor creates
the output directory. Pull the exact plan image explicitly after authenticating
to the private registry, then repeat the same command with `--execute` on the GPU
host. Use a fresh output directory for each run. Keep it inside a supervised,
bounded compute session with verified backup and provider shutdown.

Execution runs the untouched official evaluator, records unassisted native
motion and audits the readback. `replay-receipt.json` records scalar metric
differences from the saved report. It verifies source/checkpoint identity and
complete coverage; it does not promise identical floating-point simulation,
bit-exact retraining or physical success. A failed grasp remains a failed grasp
when reproduced.

## Validation status

Six CPU controls pass, including altered input/checkpoint rejection, output
isolation, full-clock binding and read-only/offline Docker arguments. The actual
166-frame e015 archive passes local preflight with its original image and
153,600-step checkpoint. The fresh-container GPU rehearsal completed on
e003/e015/e038 using relocated private packs. All full-clock/unassisted/input
contracts passed, 190 result files were hash-verified, and Brev was confirmed
STOPPED. [Metric comparisons](scored-replay-rehearsal-20261018.json) vary,
including e003 AUC 0.1478 versus 0.1818. Exact metrics and another teammate's
reproduction have not been established. The variability's cause is not isolated.

## Private portable input folder

`export_scored_replay_pack.py` accepts the same six paths plus `--code-commit`
with the complete committed replay source SHA. Its output is a new private
folder containing only the declared frozen replay assets, selected checkpoint,
producer/evaluator receipts, a relative-path recipe and per-file hashes. It
rejects non-replay data such as footage or inference weights. Keep the folder
local or transfer it through an approved private channel; no publishing occurs.

All 20 selected packs now exist privately, totalling 383.50 MB, and pass relocated
preflight and file-hash checks. e015's pack contains 19.33 MB. Three packs have
also completed the GPU rehearsal above. Independent teammate verification
remains a separate check.
