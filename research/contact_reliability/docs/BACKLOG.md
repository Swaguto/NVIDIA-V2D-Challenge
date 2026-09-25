# Team execution backlog

The live Google Doc now supplies ten names and existing workstream assignments;
see PROJECT_CONTEXT.md for the exact roster. The role-based milestones below remain
valid, but individual accountable leads and reviewers are not yet recorded. Recommended
mapping: video-prior group to reconstruction; Thanoan to contact priors/reliability;
Sinray to retargeting; Shao/Rohith/Evan to simulation; Ronnie to compute. Sharon is
currently unassigned; evaluation/reproducibility is a recommendation, not a confirmed
assignment. Each row has an accountable role, reviewer, artifact and acceptance gate.

| Due | Work item | Owner | Reviewer | Acceptance artifact |
|---|---|---|---|---|
| Sept 24 | Verify rules, access, submission contract | Project lead | Evaluation lead | Completed dependency register with official links |
| Sept 24 | Provision compatible host and assets | Infrastructure lead | Robot-learning lead | Doctor report, approved asset inventory, container versions |
| Sept 24 | Reconstruct one permitted example | Reconstruction lead | Retargeting lead | Mesh, trajectory and diagnostics with recorded units |
| Oct 1 | Load motion and run physics training | Retargeting lead | Robot-learning lead | Recorded rollout and mapping checks, not only kinematic replay |
| Oct 1 | Cost profile and experiment budget | Infrastructure lead | Project lead | Two sequence profiles, capacity forecast, 20% reserve |
| Oct 1 | Baseline evaluator dry run | Evaluation lead | Project lead | Official artifact completeness and evaluator log |
| Oct 8 | Export native reconstruction diagnostics | Reconstruction lead | Reliability lead | Validated diagnostic NPZ and sidecar on motion timestamps |
| Oct 8 | Reliability behavior and corruption checks | Reliability lead | Evaluation lead | Real development-data diagnostics; masks and missing signals reviewed |
| Oct 8 | Simulator integration smoke | Robot-learning lead | Retargeting lead | All-one equivalence and weighted run on approved task |
| Oct 15 | Four-sequence pilot | Robot-learning lead | Evaluation lead | Paired results and automated continuation decision |
| Oct 23 | Freeze method and local holdout study | Evaluation lead | Reliability lead | Config hashes, seeds, ablations and failures |
| Oct 30 | Full coverage and report | Project lead | Evaluation lead | All required sequences, honest report, runnable reproduction instructions |
| Nov 2 | Independent reproduction and submission | Project lead | Second team member | Verified archive and official acceptance receipt |
| Nov 3–4 | Correction buffer | Project lead | Relevant workstream lead | Critical fixes only; new receipt if resubmitted |

With ten listed people, explicitly cover evaluation/report writing and independent
reproduction without assuming all members have confirmed availability. Use short asynchronous updates and one weekly
integration meeting. Discuss blocked dependencies early; an owner must not label a
synthetic or replay-only result as a trained policy.

On October 1, if compute is insufficient, reduce experiment breadth and preserve
baseline task coverage. On October 15, if the pilot fails, submit the baseline and
report the analysis; do not replace the project with a new architecture.
