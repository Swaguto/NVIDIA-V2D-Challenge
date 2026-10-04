The camera-calibration fixture is the public Track 3 factual calibration JSON from
`nvidia/video_to_data_challenge`, revision
`5f68335f3acc802033d1e80728c1633197521de8`, path
`track_3/public/meta/camera_calibration.json`.

It supports the real-baseline rectification boundary test without depending on
an author's sibling data directory. Official pose-wrapper checks use the separate,
hash-verified participant kit fetched by `scripts/fetch_issue5_test_inputs.py`.
Set `V2D_SUBMISSION_KIT` to its extracted `v2d_submission_kit` directory.

Without the participant kit, parity/wrapper tests explicitly skip; a reduced test
run does not establish official scorer fidelity. The recorded validation has
109 passed, zero failed, zero skipped. No dataset RGB or mesh binaries are vendored.
