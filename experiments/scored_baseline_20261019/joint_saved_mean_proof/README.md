# Saved all-tie mean/product rounding proof (code only)

This construction addresses the actual numerical-envelope prerequisite exposed by the stopped joint method. It reads saved binary inputs only. It does not run a candidate, geometry, source, native, optimizer, old model, old control or old method query. Old SEARCH coefficients, old envelopes and their thirty one-ULP defects remain unchanged. Missing numeric dot/mean intermediate returns are not reconstructed.

## Conditional exact bound

Let `u=2^-53`, `eta=2^-1074` and `gamma_m=m*u/(1-m*u)`. All arithmetic in this proof is rational. The condition is finite RN scalar operations without overflow, with no more than the declared operation count; no installed NumPy/BLAS/compiler binary identity or upstream source accuracy is inferred.

For an exact dot of `d` saved-binary products, `m=2*d+16` and the absolute error allowance is `gamma_m*sum(abs(products)) + m*eta/(1-m*u)`. For a mean of `n` ideal terms with average incoming allowance `a`, `m=2*n+16`, the mean allowance is `a + gamma_m*(mean(abs(ideal_terms))+a) + m*eta/(1-m*u)`. Absolute subnormal allowances remain explicit. Any association within the stated RN count is covered conditionally.

For both sides' all-tie means `mu_A,mu_B` and allowances `e_A,e_B`, the final-addition allowance is `Gamma=e_A+e_B+u*(abs(mu_A)+abs(mu_B)+e_A+e_B)+eta`. The new proof envelope is `max_ALL(abs(A_ties))+max_ALL(abs(B_ties))+Gamma`. Every saved coefficient must satisfy both `abs(J-ideal_mean)<=Gamma` and `abs(J)<=envelope`. A 96-bit outward dyadic encoding encloses the rational envelope. These are numerical construction bounds, not candidate tolerances or new acceptance gates.

Command slopes use every saved factor tie. Continuous direction slopes use exact three-product proof expressions on every saved direction tie and current saved chart differential. Those ideal proof expressions are not recovered unsaved numeric matmul returns. Original source/depth/domain and phase coefficients retain exact saved-binary absolute-value copy scope, with no new source-accuracy claim.

## Complete scope and preservation

All 933560 stored dynamic coefficient positions and all 249603 pair scopes receive new proof routes, independently bound before the single bounded certificate evaluation. Conditional filtered pair rows remain typed zero-incidence records. Original 4810140 rows, all cones, strict flags, objective and conditioning remain unchanged by hash. Serialization chunks preserve all exact rational terms and are not new source partitions or row banks.

New controls cover actual synthetic mean rounding, a negative old-envelope control, correlated ties/cancellation, underflow, product/mean composition, outward encoding, unknown empty ties and typed/rational packet bits. The independent audit uses IEEE bit-decoded fixed integer accumulators at 2^-1074 and 2^-2148 rather than the producer's Fraction-from-float dot/mean algorithm. It checks every exact term and unchanged raw coefficient/envelope bytes using saved data only.

The certificate has a fixed 180-second limit and the independent audit 210 seconds, each inside the unchanged 240-second/2-GiB/1.5-GiB-reserve CPU guard. No retry, tuning, old-method continuation, reference promotion or physical qualification follows from proof success. Actual source, URDF/motion, training/integer-cell/strict-depth, fullpixel-heldout POST, native hand/object/support AND self, full swept, motor/grasp and task gates remain. F and separate unqualified components remain unchanged. Owner intent remains active and unachieved. Production runtime and tests are unchanged.

## Saved result

Seven serial CPU guards finish successfully. Eleven new controls pass (eight bound/reader controls, one exact-packet control, two independent IEEE controls). The sole certificate evaluation covers all 933560 entries and 151 frames with zero failures in 101.7850 seconds (164134912-byte observed peak RSS). The independent integer audit verifies all 3734240 exact terms in 84.0167 seconds (158089216-byte observed peak RSS). The saved mean/product discrepancy is at most 4.573361891211582e-16 in derivative algebra units. This is a different quantity from the retained thirty old-envelope defects, whose maximum gap remains one ULP/5.551115123125783e-17. No raw J/E is repaired or overwritten. No trajectory, reference, native clearance, grasp, task, policy or score qualifies.
