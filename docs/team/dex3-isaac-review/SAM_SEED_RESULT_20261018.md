# Verified SAM seed experiment: e015

The controlled experiment preserves the original two reconstructed meshes,
human fits, cameras, source clock, measured support plane and frozen collision
assets. The changed factor is retaining the verified raw SAM pose at source
frame zero instead of refining that initialization with FoundationPose. Whole
trajectories are selected using the existing source-fit selector. All original
measured-depth and temporal jump gates remain enabled.

All six seed checks passed: two objects across the three pose hypotheses retain
the exact source pose and bind it to the verified source manifest. Raw source
geometry and source confidence bytes are unchanged. The original failed
experiment and its incorrectly specified private method-label check are kept.
The recovery reused the completed control tracking instead of repeating it.

| Check | Original control | Seed-preserving experiment |
| --- | ---: | ---: |
| Initial native object/support penetration | 15.20 mm maximum | No contact |
| Training | Nonfinite metrics at about 78 iterations | 100 iterations complete |
| Environment steps | 119,808 before failure | 153,600 |
| Full unassisted native MP-SR | No completed evaluation | 0 |

The experiment's native local ADD AUC is 0.54405 and mean object position error
is 0.12886 m. Later rack/table reference penetration still reaches 24.34 mm.
These scores use changed reconstructed object references and cannot establish
hidden ground-truth accuracy or a general improvement in AUC.

This repairs initialization and training stability in this pilot. It does not
solve the physical grasp, clear the complete reference trajectory, validate the
method across episodes or justify longer PPO by itself. Keep the arm optional.

The guarded session retained 5,834 SHA-verified files and confirmed provider
STOPPED at 2026-10-18 12:32:45 UTC after 502 seconds. No private assets were
published. The standard episode contract and untouched official trajectory
reader pass on the complete eight-world output.
