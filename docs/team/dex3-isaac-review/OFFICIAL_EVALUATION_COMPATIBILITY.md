# Official evaluation compatibility — 2026-10-16

The current downloaded submission kit has SHA-256
`8d2bce64ecab6da4eee938fe2859bd60e767535af30068b386e2dc20a72cf43e`.
It is identical to the previously inspected October 15 kit. Its manifest pins
FlashCHORD to `8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef`.

Source: [official challenge and kit](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/).
The landing page describes `eval_e2e.py`; the downloaded kit's actual Track 3
entry point is `examples/run_policy_evaluation.py`, followed by
`tools/pack_submission.py --track track_3`. Use the downloaded kit's instructions.

## Verified development checkpoint

Scope: the wrapper checks below cover the archived three-hook extension and its
matching full-task checkpoints. The newest multi-contact image adds a fourth
solver dispatch hook. Its installed loader and actual physics have been checked,
but a fresh checkpoint from that image has not been tested through the kit
wrapper. See [current results and matching images](ORDERED_FIXES.md).

All 54 kit manifest members, the runtime's evaluator source, the saved checkpoint,
its complete private input manifest and frozen collision geometry passed the
new offline `scripts/dex3_training/audit_official_contract.py` check.

The untouched **kit batch runner and its included evaluator** then loaded the
existing 1,000-iteration PPO checkpoint inside the pinned extension container
and wrote `episode_000002.json` and `episode_000002.parquet` with four worlds.
This establishes this checkpoint's evaluator path in that container. It does
not establish task success: success remained zero. Raw output transfer and
hash verification are recorded separately in the compute receipts.

The required extension supplies real Dex3 geometry, the private reference schema,
and finite support geometry. The official runner composes the saved checkpoint's
configuration, including those extension targets. An unmodified stock runtime
cannot load these inputs; the complete extension image and private inputs are
part of reproduction. No evaluator or score threshold was patched.

## Development data and scored data

The historical pilot covers 75 of e002's 343 frames. The later controlled
full-task experiment covers all 343 frames and has separate initial/trained
kit-evaluator outputs with four worlds each. Both have zero success; see
[the measured comparison](full-task-policy-comparison-v1.json). e002 is not a scored episode.
The current scored roster is:

`1, 3, 5, 6, 8, 10, 14, 15, 16, 19, 20, 25, 28, 30, 32, 33, 35, 36, 37, 38`.

It requires 20 sequence policies and 36 correctly named reconstructed object
meshes, in the same body frames and metre units as their achieved rollout poses.
The twelve existing Isaac reference previews are development episodes. Completing
those previews does not cover the scored roster.

A valid final package must pass the current kit's complete frame/object roster,
trajectory/provenance and mesh checks. The official combined packer validates
these and produces the six-metric Parquet. Hidden-ground-truth scores cannot be
computed locally. No submission or Kaggle upload has been made by this work.

## Physical checks and training transfer

The latest accepted development replay remains the archived source43/seed22
Isaac diagnostic until a new full task passes its controls. An extra local
engineering diagnostic now checks relative drift within 10 mm **and** 5 degrees,
with at least 95% measured support. This is separate from the existing position
retention diagnostic and the official success metrics.

The new full-task reference adapter retains all original frames and source
confidence masks. Inferred robot commands are explicitly marked. Its action
extension adds finger feedforward inside the native capped drive target, preserving
geometric reference poses and policy residual semantics. Later extensions add
bounded measured-object acquisition, reference velocity and optional robot wrist
payload compensation. These do not make the two engines' controllers or contact
response identical. Newton retention, native actuator
readback, learned-policy improvement, other-episode validation and final packaging
remain independent acceptance gates.
