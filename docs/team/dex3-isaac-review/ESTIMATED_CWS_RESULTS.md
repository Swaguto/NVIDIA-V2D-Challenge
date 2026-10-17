# Estimated CWS policy comparison

Full-task frame-zero replay; four worlds, all 343 control steps verified with zero assistance. e002 is outside the scored roster. One seed is not robust evidence of improvement.

| Policy | New steps / parent steps | AUC | MP-SR | MPPE cm | Lift mm | Slip mm | Rotation | Pad support | Sampled depth mm | Grasp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Assisted control (no estimated CWS) | 16,384,000 / 0 | 0.7443 | 0.0% | 4.722 | 101.6–105.9 | 74.9 | 42.8° | 85.2% | 3.29 | FAIL |
| Estimated CWS + decaying training assistance | 16,384,000 / 0 | 0.8701 | 100.0% | 2.661 | 111.3–114.4 | 33.2 | 25.2° | 99.5% | 3.22 | FAIL |
| CWS actor continuation, zero training assistance | 12,288,000 / 16,384,000 | 0.8098 | 75.0% | 4.633 | 130.6–132.3 | 59.2 | 36.3° | 98.9% | 1.43 | FAIL |
| Matched zero-assistance continuation, retention weight 8 | 12,288,000 / 16,384,000 | 0.7792 | 75.0% | 3.657 | 98.9–127.0 | 60.4 | 31.8° | 62.3% | 3.17 | FAIL |
| Complete PPO-state continuation, original training recipe | 12,288,000 / 16,384,000 | 0.7733 | 0.0% | 3.486 | 111.8–116.7 | 42.1 | 26.1° | 97.8% | 1.55 | FAIL |

The estimated-CWS run started with exactly the same actor tensors as the assisted control. Parent and new-run step counts are recorded separately. Source observations, geometry, mass, gravity, friction, motor caps, action limits and quality gates were not relaxed.

[All per-world measurements and hashes](estimated-cws-policy-comparison-v1.json) · [guidance method and reproduction](ESTIMATED_CONTACT_GUIDANCE.md)
