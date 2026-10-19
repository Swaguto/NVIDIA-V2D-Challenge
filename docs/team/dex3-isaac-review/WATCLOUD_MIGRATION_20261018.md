# Watcloud source-hand recovery checkpoint

Watcloud is now the selected compute provider. Brev remains stopped and is not
being retried during this migration. This checkpoint verifies the perception
runtime and returns new estimated hand priors; it does not establish a physical
grasp, a policy improvement or a hidden evaluation score.

## Measured runtime

| Item | Actual observation |
| --- | --- |
| Connection | Owner's existing `watcloud` SSH alias |
| Scheduler | Slurm under `/opt/slurm/bin`; compute runs inside allocations |
| Compute node | `trpro-slurm1`, RTX 3090 24 GiB, driver `580.178.04` |
| Container | Existing perception digest `6e2039d0a54e14a188a1798efe57c3465d002f063c782c827b791e4418d2dce6` |
| Source overlay | `60c943f0c50af779a11ca7ff79ed8642fd966e59` |
| Actual container CUDA probe | PyTorch `2.5.1+cu124`; RTX 3090, capability 8.6; device tensor readback passed |
| Setup probe | Job `711097`, completed, exit `0:0`, four seconds |
| Eight-image inference | Job `711098`, completed, exit `0:0`, 10 minutes 59 seconds including transfer/setup |
| Source inference interval | Approximately 27 seconds, separately from setup/transfer |
| Private result backup | 29 files independently checked against the worker's hashes on the owner's Mac |
| Allocation after backup | Completed; subsequent owner queue read was empty |

The inference job requested four CPUs, 24 GiB host RAM, 20 GiB GPU shards and
100 GiB temporary disk, with a 45-minute scheduler wall limit and an earlier
controller timeout. Only one GPU allocation ran at a time. Model inference had
its own 900-second cap. The inference container had networking disabled; source
code, inputs and weights were separate read-only mounts. Registry authentication
was transient within the job's private scratch and removed after the image pull.

The owner's home usage was 20.36 GiB against a 20 GiB quota. Models, container
layers and inputs were staged in allocated scratch. Large persistent storage
under the owner's private NAS/project path is still unresolved. Most of this
first job's time was a 3.608 GB private transfer; retaining a verified model cache
on suitable private storage will avoid repeating that transfer in later jobs.

The cluster's [Slurm guide](https://cloud.watonomous.ca/docs/compute-cluster/slurm)
documents shard allocations, temporary disk and rootless Docker. Its
[storage guide](https://cloud.watonomous.ca/docs/compute-cluster/user-manual)
and [quotas](https://cloud.watonomous.ca/docs/compute-cluster/quotas) explain why
large model caches belong outside the home directory. Login nodes are used for
submission and lightweight control.

## Quality result and retained failures

SAM2 and HaMeR produced estimated priors for all eight requested e037 frames.
The original hand records, subject shape, scale, confidence and missing flags
remain unchanged. The checked CPU merger stages new priors in a separate,
explicitly unqualified handoff.

At carry frame 50, median held-out error against the existing inferred image
tracks decreased from **307.98 px** for the old prior to **19.42 px** for the
fresh, unfitted prior. The first bounded source fit reached **7.01 px** median
and **20.70 px** p95, but its estimated joint-depth error remained **35.79 mm**,
above the unchanged 10 mm source screen. No frame passed that complete screen.
Frames 40 and 55 lacked enough independent image/depth constraints for fitting.

The diagnostics expose two additional problems:

- Inferred joint centres may lie behind the visible skin surface. Sampling skin
  depth and adding a constant 5 mm does not establish the depth of an occluded
  finger joint. The failed joint-depth control remains preserved.
- Hand and spoon masks conflict: at frame 50, **4,459 of 18,780** proposed hand
  pixels also belong to the independently inferred spoon mask. Both masks are
  estimates; disagreement is treated as unknown, not a new hand/contact label.

A new dense-skin control used object-conflict exclusion and a frozen, inferred
wrist boundary. With 32 training and 32 held-out skin pixels, its first fit hit
all 32 held-out rays and achieved **1.64 mm** median / **6.25 mm** p95 surface
depth error, but image p95 was **42.10 px**, failing the 25 px screen. Normalizing
the skin channel for sample count improved image median to **6.77 px**, but p95
remained **34.31 px** and only 31 of 32 held-out rays hit the mesh. That control
also remains rejected. Its sample-count invariance control passed.

These are source reconstruction experiments, with estimated stereo depth and
inferred masks/tracks. No source confidence was upgraded, no numeric gate was
relaxed, no robot retargeting or physics was rerun, and no PPO steps were added.

## Next work

1. Resolve hand/object mask ambiguity and visibility of the inferred finger
   landmarks; preserve unknown regions and the failed controls.
2. Validate source geometry across the carry interval before expanding the new
   source method. Do not spend on another broad policy cohort yet.
3. Retarget a source-qualified candidate under the original morphology and actual
   URDF limits. Require the unchanged **1 mm full native reference-contact gate**,
   motion, swept-contact and grasp-proximity checks before any PPO.
4. Capture passive startup traces only during a justified new physical candidate
   run; do not repeat completed replay jobs to investigate variability.

The exact executed experiment sources are archived under
`experiments/scored_baseline_20261018/watcloud/` with hashes and execution labels.
They retain the recorded private workspace assumptions and are not a generic
fresh-host reproduction CLI. Private model files, footage, reconstruction packs
and results are not included. Independent teammate reproduction remains
unverified.
