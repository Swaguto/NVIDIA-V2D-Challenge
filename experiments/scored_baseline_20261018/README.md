# Archived measured seed producer

`metric_seed_candidate_v3.py` is the exact CPU producer archived with the
measured-surface e003 candidate. Its SHA-256 is
`5d47bb42befc766dff7a3de385dc2f97342d1e25259b1bf405438a1109a6452d`.
The original machine path is retained as historical source; no inputs, footage,
weights or credentials are included here.

Use `scripts/track3/reproduce_metric_seed_candidate.py` from the checkout to run
those exact numerical functions with portable imports. It verifies the archived
source hash and the private original SAM input closure. Supply the original
private support/camera bundle and optional warm-start candidate report when
repeating a historical run. The wrapper records actual library versions.

This fits only uniform scale and translation from measured foreground. Rotation
and the raw body origin remain fixed. The historical plane constraint applies
only to a seed independently known to lie over the table. It has no general
off-table classifier, so that mode requires an explicit scope assertion. The
newer finite-footprint translation experiment is a different method.

Some archived fits regress; fitting does not promote a candidate. Preserve the
candidate report, original bytes and optimizer failures. The calibrated-seed
verifier and native/full-task checks remain necessary before use. Small numerical
differences across platforms/libraries can change optimization outcomes. The
frozen private policy replay pack is the route for repeating an exact archived
policy, without rerunning reconstruction or inference weights.
