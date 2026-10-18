"""Do not train a quality candidate on clipped, stale or colliding references."""

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from reference_contact_gate import assess


@pytest.fixture
def audits(tmp_path):
    manifest = dict(source_frame_indices=[0, 1, 2], objects=[dict(name="spoon")])
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    common = dict(
        input_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        reference_knots=3,
        physics_stepped=False,
        source_targets_modified=False,
        native_mesh_checks_relaxed=False,
        reporting_tolerance_m=0.001,
        runtime_commit="pinned-runtime",
    )
    objects = dict(
        common,
        schema="v2d.native-reference-object-contact-audit.v1",
        objects=["spoon"],
        reference_object_poses_verified=True,
        frames=[dict(reference_frame=f, contact_pairs=[]) for f in range(3)],
    )
    hands = dict(
        common,
        schema="v2d.native-reference-hand-contact-audit.v1",
        reference_robot_and_object_poses_verified=True,
        all_native_hand_body_fk_verified=True,
        frozen_source_closure_verified=True,
        frames=[dict(reference_frame=f, contacts=[]) for f in range(3)],
    )
    return path, manifest, objects, hands


def test_clear_reference_passes_only_the_narrow_gate(audits):
    r = assess(*audits)
    assert r["native_penetration_gate_passed"] is True
    assert (
        r["physical_task_accepted"]
        is r["contact_forces_or_grasp_stability_assessed"]
        is False
    )


@pytest.mark.parametrize("kind", ["hand", "object"])
def test_late_contacts_are_recomputed_even_if_summary_claims_clear(audits, kind):
    path, m, o, h = audits
    o["summary"] = h["summary"] = []
    if kind == "hand":
        h["frames"][2]["contacts"] = [
            dict(hand="right", target="spoon", signed_gap_m=-0.012)
        ]
    else:
        o["frames"][2]["contact_pairs"] = [
            dict(bodies=["support", "spoon"], maximum_penetration_m=0.012)
        ]
    assert assess(path, m, o, h)["native_penetration_gate_passed"] is False


@pytest.mark.parametrize(
    "change",
    [
        "truncated",
        "reordered",
        "stale",
        "unchecked-fk",
        "modified-targets",
        "runtime",
        "nan",
        "unknown-target",
    ],
)
def test_partial_or_unbound_measurement_fails_closed(audits, change):
    path, m, o, h = copy.deepcopy(audits)
    if change == "truncated":
        h["frames"].pop()
    elif change == "reordered":
        h["frames"].reverse()
    elif change == "stale":
        h["input_manifest_sha256"] = "0" * 64
    elif change == "unchecked-fk":
        h["all_native_hand_body_fk_verified"] = False
    elif change == "modified-targets":
        h["source_targets_modified"] = True
    elif change == "runtime":
        h["runtime_commit"] = "different"
    else:
        h["frames"][1]["contacts"] = [
            dict(
                hand="left",
                target="spoon" if change == "nan" else "unrelated",
                signed_gap_m=float("nan") if change == "nan" else 0,
            )
        ]
    with pytest.raises(ValueError):
        assess(path, m, o, h)
