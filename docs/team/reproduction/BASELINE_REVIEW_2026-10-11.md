# Separate review of the e012 reproduction branch

The source reviewed was `a70d2efbad12100b341057535f4699058d2b6364`, published as
`recent-reproducibility-code`. The automatic hand-recovery candidate is still
rejected for placement accuracy. Its existing holdout evidence remains frozen;
neither its inference settings nor any saved measurement/replay pack changed.

The continuous presentation layer keeps held poses separate from source
observations. Review found that verification did not enforce the manifest's
accuracy/contact/training flags and support counts. The repair refuses a
manifest claiming acceptance or misreporting held/observed counts, even if
the payload hashes and arrays still match. The existing4 continuity/provenance
tests exercise these failure cases; all14 reproduction controls pass.

No new inference, optimizer, IK or simulator experiment is claimed by this
metadata repair. The recorded181-frame e012 replay and source masks stay
unchanged. This branch is reviewed separately from the e002 Dex3 baseline in
PR35, and remains unsuitable as an accepted hand/contact training method.
