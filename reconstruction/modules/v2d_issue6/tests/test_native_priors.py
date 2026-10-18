import json

import pytest

from issue6.native_priors import load_native_priors


def record(folder, name, frame, right=False):
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / name
    p.write_text(json.dumps({"mano": {"hand_pose": [1, 2]}, "frame_idx": frame, "is_right": right}))
    return p


def test_explicit_conversion_path_preserves_cropped_source_axis(tmp_path):
    folder = tmp_path / "separate-conversion/converted"
    p = record(folder, "first.json", 0)
    record(folder, "last.json", 1, True)
    before = p.read_bytes()
    rows = load_native_priors(folder, [{"id": 3}, {"id": 7}])
    assert list(rows["left"]) == [3]
    assert list(rows["right"]) == [7]
    assert rows["left"][3]["inference_frame_idx"] == 0
    assert p.read_bytes() == before


def test_canonical_source_indices_are_not_remapped(tmp_path):
    record(tmp_path, "canonical.json", 7)
    rows = load_native_priors(tmp_path, [{"id": 3}, {"id": 7}], source_frame_indices=True)
    assert list(rows["left"]) == [7]
    assert "inference_frame_idx" not in rows["left"][7]


def test_missing_or_empty_prior_cannot_silently_erase_hands(tmp_path):
    with pytest.raises(ValueError, match="unavailable"):
        load_native_priors(tmp_path / "missing", [{"id": 3}])
    with pytest.raises(ValueError, match="No native hand poses"):
        load_native_priors(tmp_path, [{"id": 3}])
    record(tmp_path, "bad-index.json", 3)
    with pytest.raises(ValueError, match="outside"):
        load_native_priors(tmp_path, [{"id": 3}])
