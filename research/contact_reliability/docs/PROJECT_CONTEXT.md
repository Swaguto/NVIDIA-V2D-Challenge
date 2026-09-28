# NVIDIA V2D baseline project context

## September 25, 2026 update (supersedes stale status below)

The live Google Doc was refreshed across all six tabs in this session. The team
now plans CHORD + stereo; the older Octo/Cosmos assignments below are historical.
The user accepted a proposed role covering CHORD baseline training (#10) and
contact robustness (#13), with 25+ hours/week. Team assignment is not confirmed.
GitHub is connected as Evc-r. Reviewed team main:
`b6dd0d9f9ddabf7f1955119d797d9ccfe8107104`.

GPU availability is team-reported: shared L40S 48 GB, built containers, zero-action
smoke videos. No remote training was run in this session. SSH connection details
and a training-ready team reference are still needed. Public pose tables and
metadata have now been downloaded locally; this is not a training bundle.
Official eval_e2e.py availability is unresolved; PR #17 is a local proxy scorer.

See [EVAN_DELIVERABLES.md](EVAN_DELIVERABLES.md) for implementation, commands,
variant semantics, evidence levels and remaining gates, and
[EVALUATION_HANDOFF.md](EVALUATION_HANDOFF.md) for source-review findings.
The user prohibits commits, pushes and PRs without explicit instruction.
Local CPU tests now run successfully, including PyTorch reward parity. No learned
policy, matched GPU ablation, official score or submission has been produced.

## Historical September 21 snapshot

User-designated living source:
https://docs.google.com/document/d/1QhZwCoRwiCLJyEz5xhOKyFkx1qLhnBljTmOwKyjzvAM/edit

Reviewed September 21, 2026 using the connected profile evanchiu01@gmail.com.
Drive reported last modification at 2026-09-21T23:56:37.394Z. All three tabs and
both embedded copies of the pipeline diagram were reviewed. No comment threads
were returned. This is a context summary, not a complete version-history audit.
The live document supersedes the older Downloads/NVIDIA V2D Challenge.docx for
current team context; direct user instructions and verified official rules remain
distinct sources of authority.

## Document tabs

1. Rules & Roles — t.bdr6a9pwkq1.
2. TRACK 03 Egocentric Egocentric Video → Policy — t.0.
3. Organizer Questions & Compute email — t.iofm04ajmmrd.

## Recorded team assignments

Preserve these spellings from the table; do not silently resolve name variants in prose.

| People | Workstream recorded in the document |
|---|---|
| Thanoan | Action Priors (Torque VLA) |
| Gurshaan, Akshay, Roshan | Video Priors (ML) |
| Shao, Rohith, Evan | Isaac Sim (Simulation work) |
| Shao, Rohith, Evan | NVIDIA Cosmos (Simulation work) |
| Shao, Rohith, Evan | OCTO integration (Simulation + ML) |
| Ronnie | Compute (important) |
| Sinray | Mapping (kinematic retargeting) |
| Sharon | Free Role |

The table contains ten distinct names. That replaces the earlier planning assumption
of seven to nine people; it does not establish ten confirmed registrations or ten
identical weekly commitments. The user previously chose 8–12 hours per member as a
planning assumption. The doc expects regular work and weekly 1–2 hour team meetings.

## Goals and constraints

- Track 3: egocentric video to robotic skills, ending in the official eval_e2e.py artifact.
- Industry Innovation Award is the preferred target; leaderboard awards are secondary.
- November 4 submission deadline; the existing internal target is November 2.
- The doc describes possible personal compute spending of $200, conditional on progress.
  Currency, firm cap, and spending authorization remain unconfirmed.
- Longer-term research/publication remains valuable if the competition is unsuccessful.
- The document is intended for the team; do not distribute it outside that context.
- The organizer email is still a draft. Its instruction to finish/send work is not an
  instruction from the current user to send a message.

## Technical intentions versus established facts

The doc explores VideoDex-style video priors, action priors, contact physics,
kinematic retargeting, Isaac Sim/Isaac Lab, Cosmos, and Octo. It sketches video priors
to a generalist policy to torque/VLA reasoning to movement. It also raises arbitrary
robot transfer and disturbance recovery as longer-term interests.

These are not a validated integrated architecture. Questions and tentative claims
about torque recovery, DDR, Open X-Embodiment, Cosmos and Octo require primary-source
verification before implementation. Do not equate gripper action encoding with
measured contact force, or model-generated torque suggestions with physical evidence.

The diagram shows human demonstration video → ingestion → reconstruction → human-object
trajectories/simulation → robotic grounding → augmentation → physics-grounded data →
foundation models → robot. It also shows a direct grounding-to-robot path. Interpret
it as the platform's capability map, not proof that every box is required for this entry.

## Comparison with the earlier attached DOCX

The main substantive addition is the populated people/roles table. The longer-term
venue list also adds ACM SIGGRAPH. The force/torque questions, VideoDex/Octo/Cosmos
ideas, award preference, provisional budget, deadline and organizer email were already
present in the earlier document. Do not describe them as newly introduced requirements.

## Effect on the current development plan

Keep the accepted reproducible V2D baseline plus contact-reliability experiment.
The updated context changes staffing and makes the scope mismatch more explicit:
the team has named Octo/Cosmos workstreams, while the current implementation intentionally
does not include them. Recommend bounded feasibility studies with a concrete downstream
metric and compute cost before promoting either into the November critical path.

Recommended mapping, not newly confirmed assignments:
- Video-prior group: reconstruct, export diagnostics, test visibility/occlusion reliability.
- Thanoan: review contact/action priors and reliability-weighted contact guidance against CHORD.
- Sinray: retargeting plus body, frame, unit and timestamp contracts.
- Shao/Rohith/Evan: reproduce the Isaac Lab baseline and validate rollout/evaluation integration;
  assess Octo/Cosmos only after an end-to-end baseline works.
- Ronnie: obtain compatible compute access and a measured budget/capacity forecast.
- Sharon: proposed evaluation/reproducibility owner; she is currently unassigned in the source.

The group still needs one accountable integration lead and an evaluation/report owner.
Do not assume the project lead or individual responsibilities within the simulation
group are confirmed simply because Evan owns this Codex task.

## Actual implementation status

Research toolkit exists in this workspace: pinned upstream source, signal extraction,
normalization, reliability sidecars, five contact-reward adapters, ablations, experiment
splits, run manifests and guarded official-evaluator invocation. Latest checks: 18 local
tests plus 17 upstream motion-schema tests passed; synthetic smoke passed. See VALIDATION.md.

Still missing: official dataset/evaluator, licensed assets, validated Linux GPU host,
native reconstruction diagnostic export validation, and real simulation/training/evaluation.
The user confirmed no separately obtained dataset and uncertainty about compute access.
Local GPU is an 8 GB RTX 5060 Laptop (sm_120); one upstream reconstruction path rejects
that architecture, and Docker was not found. No policy or official submission exists.
