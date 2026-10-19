# Coupled index source and time algebra

Three serial CPU jobs derive and independently check continuous half-angle
constraints across frames 107–111. This archive contains exact executed
code, guards and plans. Source arrays, coefficients, journals and results
remain private. Production runtime and its tests are unchanged.

For actual angles q_j = 2 atan(u_j), define
P = product(1 + sign(k_j) i u_j)^abs(k_j).
Its squared modulus is product(1 + u_j^2)^abs(k_j), always at least one.
On a certified angle branch, L <= sum(k_j q_j) <= U is equivalent to two
polynomial sine-sign inequalities. This does not assume that the tangent
of an adjacent difference has a positive denominator.

The new system binds five saved source polynomials and 22 temporal
constraints (44 polynomial inequalities) to one correlated outer region.
Original adjacent/second-added limits and actual timestamp velocities
remain unchanged. Rational Taylor intervals enclose sine/cosine constants
for saved binary inputs. Outward endpoint/arithmetic conversion guards are
necessary relaxations only, never candidate acceptance tolerances.

Twelve new analytic contracts pass. Independent checks validate 272
coefficient enclosures, all 22 angle branches, and 702 direct angle/timestamp
test vectors; 348 intentionally rejected vectors are retained. Maximum
phase closure is 2.50e-16. These vectors are algebra tests, not source/native
or path candidates. Three unknown source-event bands remain explicit.

No feasibility solve, whole-clock trajectory, exclusion proof, native position,
physics, GPU/PPO or score success follows. The verified reference and separate
unqualified middle component remain unchanged. No old producer or study reran.
