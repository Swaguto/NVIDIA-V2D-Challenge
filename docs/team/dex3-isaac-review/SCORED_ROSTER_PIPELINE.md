# Preparing a fresh scored-roster pipeline

`scripts/track3/make_roster_configs.py` writes an ordered hands → reconstruction
→ policy plan for every episode in the official kit. It checks the kit's sample
frame axes, object order, three source-video paths per episode, code contexts,
robot asset lock, private model directory layout and immutable image digests.
It does not start Docker, train, allocate cloud compute or load weights.

For exact playback of an existing learner checkpoint, use
[SCORED_POLICY_REPLAY.md](SCORED_POLICY_REPLAY.md). Fresh reconstruction and
training can differ numerically. This uniform baseline also does not recreate
the separately selected e003 geometry or the e015 seed experiment; replay packs
retain those selected inputs. Completion of this pipeline is not a grasp pass.

## Prerequisites

Use a Linux NVIDIA host with Docker and the NVIDIA container runtime. Obtain
private inputs directly from the owner; no footage, MANO weights or other gated
weights are distributed in Git. Prepare code contexts using the committed
context builders and robot assets using the file lock. Read the context/image
setup in [SCORED_BASELINE.md](SCORED_BASELINE.md).

Create a private settings JSON using absolute paths on the machine that will run
the stages:

```json
{
  "source_root": "/data/video-only-source",
  "data_subdir": "track_3/evaluation",
  "models": "/data/private-models",
  "perception_context": "/data/perception-context",
  "policy_context": "/data/policy-context",
  "robot_assets": "/data/robot-assets",
  "robot_asset_lock": "/data/robot-assets.lock.json",
  "kit": "/data/v2d_submission_kit",
  "work_root": "/data/fresh-roster-run"
}
```

Images default to the committed immutable runtime digests. Optional settings
`perception_image`, `sam3d_image`, and `training_image` must also be immutable.
`docker_host` applies to the perception stages; policy execution uses the host's
default Docker daemon. Keep credentials out of this file.

```bash
python scripts/track3/make_roster_configs.py \
  --settings /data/private-settings.json --out /data/fresh-roster-plan
```

The default selects the complete official roster. `--episodes 3 15 38` selects
an explicit subset while retaining the full roster count in the plan. Existing
episode outputs are rejected. The plan records configuration hashes, exact code
commits and stage commands. Commands invoke the verified runners inside their
own pinned contexts. If relocating to another host, regenerate the plan against
that host's verified paths and Python environment.

## Running and assessing the plan

Use the three stage commands for each episode in their recorded order. A later
stage's runtime preflight requires the preceding stage's actual outputs; plan
generation alone does not certify those outputs. Each runner has a time limit.
A surrounding compute controller must enforce the total lease, hash-verify
backups, and confirm provider shutdown whenever work stops. This planner is not
a cloud scheduler or a spending cap.

Keep failed cases and original controls. Check source projections, initial and
full-reference contacts, finite native rollouts, and independent evaluation
before accepting a policy. Fresh policy contexts require the full hand-contact
diagnostic; older archived contexts retain their historical behavior. Do not
interpret local scores against inferred references as hidden leaderboard scores.

After all required outputs pass the submission contracts, use
`prepare_scored_submission.py` with the official kit and an explicit episode
catalog. It invokes the untouched packer. A packed artifact still requires the
captain's review and approval before submission.
