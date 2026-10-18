# Reset-settling confirmation — 18 October 2026

**Keep checkpoint 2000.** Short reset settling has not qualified as a replacement.
All four continuations completed 100 iterations / 819,200 environment steps each.
Each checkpoint received three complete, unassisted, eight-world native
readbacks under the unchanged evaluation recipe.

| Exploration fork | Training reset | Mean AUC | Mean MP-SR | Mean object position error | Mean relative position error |
| --- | --- | ---: | ---: | ---: | ---: |
| 101 | Control, 20 settling steps | 0.86067 | 91.67% | 0.01792 m | 4.927 cm |
| 101 | Short, 5 settling steps | 0.86173 | 95.83% | 0.01833 m | 5.112 cm |
| 202 | Control, 20 settling steps | 0.85168 | 87.50% | 0.24135 m | 49.613 cm |
| 202 | Short, 5 settling steps | 0.86087 | 91.67% | 0.01793 m | 5.009 cm |

The paired AUC changes are +0.00107 and +0.00919. Individual repeated readbacks
span approximately 0.03 AUC in some cases. The previously retained parent's
three-readback mean is approximately 0.86549 AUC. The short-settling candidates
therefore do not establish a repeatable AUC improvement over that parent.
Fork 101 also slightly worsens position and relative-position errors. Fork 202
avoids a large-error control repeat; this is useful stability evidence, but does
not identify reset settling as the complete cause of the original regression.

The fork changes only the learner's exploration PRNG key. All other 59 learner
arrays and counters remain byte-identical before continuation. The environment
reset seed remains 42. These are independent exploration streams from one parent,
not fresh policy initializations. File ownership was repaired for the two private
fork files after verifying their SHA256 values; their tensors did not change.

Parent checkpoint SHA256:
`4d91fd4164b043a41a390ee7fbb904a3f3aa6a0bb724e19fc1f2342e3f108a40`.

Total new training: **3,276,800 environment steps**. Every native physical task
audit still fails. These reconstructed-reference diagnostics are not hidden
leaderboard scores. No automatic longer continuation or checkpoint promotion
follows from this result. Relative-object reward balance remains a separate,
untested hypothesis. The scored baseline already uses zero settling; this legacy
5-versus-20 comparison must not be presented as a validated scored-task fix.
