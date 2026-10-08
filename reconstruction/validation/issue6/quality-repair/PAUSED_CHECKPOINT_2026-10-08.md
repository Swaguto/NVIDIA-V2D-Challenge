# Issue 6 paused checkpoint — October 8, 2026

Issue 6 is incomplete. Final acceptance is **0/20**; this checkpoint does not
promote a reconstruction or establish contact ground truth. Keep the issue open
and PR32 in draft. The user requested finishing the current episode 22 run,
pausing, then pushing this checkpoint. The authorized cumulative compute/storage
cap is US$40.

## Episode 22: completed diagnostic trial 31

Six stages (validate, hands, depth, align, contacts, report), contact/geometry/
stage audits and review rendering completed at 2026-10-08T20:44:03Z. Two additional
source-reviewed current-frame fits at frames 88 and 89 were integrated into
trial 30. Baseline records and other masks were preserved. The contact threshold,
source visibility denominator and zero-bridging setting were unchanged.

| Visible grasp | Usable geometry | Strict raw contact evidence | Longest unsupported gap | Numerical gate |
| --- | ---: | ---: | ---: | --- |
| Left cup | 97.12% | 94.24% | 0.099 s | Pass |
| Right pitcher | 87.50% | 73.96% | 0.296 s | **Fail** |

The right-hand criteria remain at least 90% usable geometry and 80% strict raw
contact evidence. Its remaining longest gap is frames 82–87. These are internal
engineering gates, not official challenge scores. Source correspondence,
penetration, release and full video review remain necessary.

Evidence bindings:

- Configuration SHA-256: `79e281c79347756f187cc30c48a698c650a37c472e41a6a97989571fe2faa996`
- Grasp audit SHA-256: `a38639e61cc4db142dde41fdbf3a4e6f23c86763151c950748cfb80f3d682dd8`
- CAD/hand review video SHA-256: `b3f00695fb189daa8fefe14c6f3cae7cc08dfc43dc5211a22340decbaea48b16`
- Private archive SHA-256: `97b5472243dad356013f14dbac570e3a16e71920c0b65e480abac950eb00038e`

The 479,775,118-byte archive passed all 3,358 member checks and 954 external-input
checks on the retained compute volume at 20:45:36Z. Its receipt and numerical
audits were downloaded locally. **The complete trial31 archive has not been
downloaded and independently checked locally; the new videos have not completed
visual review.** Private footage, weights and licensed MANO assets are not in Git.

## Resume work

1. Retrieve and independently verify the episode 22 archive, then review its
   exact videos and remaining grasp failures.
2. Repair episode 12's early finger geometry and episode 41's remaining failure;
   complete pending release/geometry reviews for the other development trials.
3. Finish the interrupted episode 39 input transfer and remaining independent
   input verifications. Preserve the five holdouts from adaptive tuning.
4. Freeze the production configuration, run the holdouts, and finish all 20
   episode receipts, source/contact audits, videos and acceptance reviews.

Episode 2's existing reviewed baseline archive was rehashed locally and is a
candidate for issue 7 joint refinement. No issue 7 refinement has been executed.
Starting that experiment does not close issue 6 or waive the all20 requirement.

The official Track3 rules refresh distinguishes final learned-policy object
trajectory and mesh evaluation from intermediate hand/contact reconstruction.
The remaining active-grasp correctness failures still matter for downstream
use; successful robot training has not been demonstrated.
