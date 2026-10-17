# Estimated CWS policy comparison

Full-task frame-zero replay; four worlds, all 343 control steps verified with zero assistance. e002 is outside the scored roster. One seed is not robust evidence of improvement.

| Policy | New steps / parent steps | AUC | MP-SR | MPPE cm | Lift mm | Slip mm | Rotation | Pad support | Grasp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Assisted control (no estimated CWS) | 16,384,000 / 0 | 0.7443 | 0.0% | 4.722 | 101.6–105.9 | 74.9 | 42.8° | 85.2% | FAIL |
| Estimated CWS + decaying training assistance | 16,384,000 / 0 | 0.8701 | 100.0% | 2.661 | 111.3–114.4 | 33.2 | 25.2° | 99.5% | FAIL |

The estimated-CWS run started with exactly the same actor tensors as the assisted control. Parent and new-run step counts are recorded separately. Source observations, geometry, mass, gravity, friction, motor caps, action limits and quality gates were not relaxed.

[All per-world measurements and hashes](estimated-cws-policy-comparison-v1.json) · [guidance method and reproduction](ESTIMATED_CONTACT_GUIDANCE.md)
