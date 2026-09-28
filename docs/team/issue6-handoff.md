# Issue 6: hand reconstruction and stereo contact integration

The local integration milestone provides separate validation, hand, depth,
alignment, contact and report stages. It is CPU-tested with synthetic hand
coordinates; real GPU inference and three-episode acceptance remain pending.

- [Setup, manifest contract and commands](../../reconstruction/modules/v2d_issue6/README.md)
- [Calibration/object/compute handoffs and reviewer handback](../../reconstruction/modules/v2d_issue6/HANDOFF.md)
- [Validation evidence and remaining gates](../../reconstruction/modules/v2d_issue6/VALIDATION.md)
- [Incomplete episode-2 manifest template](../../reconstruction/modules/v2d_issue6/examples/episode2.template.json)

Episode 2 is the first 64-frame smoke run, followed by the complete episode;
episodes 12 and 13 require their own bundles. Preserve the proxy holdout split.
GT-assisted calibration provenance must remain explicit even with predicted
object poses. This contribution does not close issue #6 or establish contact
precision, robot training, or a competition improvement.
