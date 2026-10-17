# Retained complete-state PPO checkpoints

Full frame-zero four-world replay; no new PPO training in this screening. Contacts are sampled at 20 Hz. The original parent has 16,384,000 steps; the continuation ends at 28,672,000.

| Global iteration | AUC | MP-SR | Pot slip mm | Pot rotation | Required-pad support | Sampled depth mm | All-grasp gates |
|---:|---:|---:|---:|---:|---:|---:|---|
| 2000 | 0.8701 | 100% | 33.2 | 25.2° | 99.5% | 3.22 | FAIL |
| 2500 | 0.8560 | 100% | 39.8 | 25.6° | 97.8% | 1.84 | FAIL |
| 3000 | 0.8180 | 75% | 41.8 | 27.1° | 99.5% | 1.64 | FAIL |
| 3500 | 0.7733 | 0% | 42.1 | 26.1° | 97.8% | 1.55 | FAIL |

Selected development checkpoint: `policy-grip-estimated-cws-2000-v2`. Selection minimizes the largest normalized engineering violation, with passing candidates preferred. It is not an official score or physical acceptance.

[Per-world evidence, hashes and lifecycle](retained-ppo-checkpoint-screen-v1.json) · [reproduce the complete-state continuation](EXACT_PPO_RESUME.md)
