# Finite compiled-hand contact reachability

Exact code only: 24 source/guard/plan files with SHA-256 bindings. Restore the
private study layout using `source-manifest.json`; private geometry, commands,
logs and reports are excluded. Historical runtime/test snapshots bind earlier
results to their actual executed sources. They are not new production modules.
Do not rerun completed jobs.

Eight guarded CPU jobs: common-translation library verification v1 (broadcasting
fixture failure), v2 (11 pass), finite atlas, facet-LP/native checks, library v3
(18 pass after whole-hull API extension), whole-hull common-root check, saved
wrench-span diagnosis, library v4 (19 pass including three-facet closure). Each
guard enforces 240 seconds, 2 GiB RSS and a 1.5 GiB disk reserve.

The atlas has 344/50/50 thumb/index/middle configurations. A frozen 32-pair queue
yields nine geometric seeds with native thumb contacts, but all fail physical
checks. The same queue under conservative whole-hull planes has no solution.
Eight two-point contact sets fail even unrestricted wrench-span membership; a
three-contact case has full rank but still fails constrained loads and has
palm/thumb self-collisions. These finite failures do not prove global task
impossibility. Old hand/object certificates explicitly do not cover self-contact.

No source truth, confidence, model parameter, root bound or gate was changed. No
physics step, old native recompile, PPO or GPU job ran. See the appended wrist
report and `scripts/dex3_training/CONTACT_PATCHES.md` for qualifications.
