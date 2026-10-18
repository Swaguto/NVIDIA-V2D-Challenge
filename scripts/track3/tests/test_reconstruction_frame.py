import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from select_reconstruction_frame import mask_features, metric_area, rank


def test_camera_distance_cannot_exclude_the_clean_complete_spoon_view():
    rows = []
    # Recorded source-only failure: pixel area excluded the clean distant view
    # while the later, closer view overlaps an observed hand by 59%.
    for frame, area, z, hand, box in [
        (0, 9247, 0.57683, 0, 0),
        (136, 20600, 0.37796, 0.58534, 0.07361),
        (200, 30000, 0.35, 0.8, 0.2),
    ]:
        mask = np.zeros((200, 200), bool)
        mask.flat[:area] = True
        measure = metric_area(
            mask, np.full(mask.shape, z), np.ones_like(mask), 565, 565
        )
        rows.append(
            dict(
                frame=frame,
                area=area,
                overlap_fraction=0,
                hand_overlap_fraction=hand,
                bbox_contamination=box,
                **measure,
            )
        )
    assert rank(rows)[0][0]["frame"] == 136
    assert rank(rows, "metric_area_proxy_m2")[0][0]["frame"] == 0
    assert rows[0]["metric_area_proxy_m2"] > rows[1]["metric_area_proxy_m2"]


def test_unknown_or_nonmetric_depth_does_not_supply_a_view_area():
    mask = np.ones((20, 20), bool)
    assert metric_area(mask, np.ones(mask.shape), mask & False, 100, 100) is None
    assert metric_area(mask, np.full(mask.shape, 1000), mask, 100, 100) is None
    assert metric_area(mask, np.full(mask.shape, np.nan), mask, 100, 100) is None
    with pytest.raises(ValueError, match="matching"):
        metric_area(mask, np.ones((2, 2)), mask, 100, 100)


@pytest.mark.parametrize("with_depth", [False, True])
def test_unknown_other_foreground_is_excluded_without_losing_known_views(
    tmp_path, monkeypatch, with_depth
):
    import json
    from types import SimpleNamespace

    import cv2
    from select_reconstruction_frame import select

    masks = tmp_path / "masks"
    for name in ["target", "other"]:
        (masks / name).mkdir(parents=True)
    (masks / "report.json").write_text(
        json.dumps({"episode": 1, "objects": ["target", "other"], "frames": 2})
    )
    target = np.full((20, 20), 255, np.uint8)
    for frame in [0, 1]:
        cv2.imwrite(str(masks / "target" / f"{frame:06d}.png"), target)
    # Frame 0 is otherwise the earliest clean candidate but its other-object
    # observation is missing. Frame 1 has explicitly observed empty foreground.
    cv2.imwrite(str(masks / "other/000001.png"), np.zeros_like(target))

    class Capture:
        def set(self, *_):
            pass

        def read(self):
            return True, np.zeros((20, 20, 3), np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: Capture())
    monkeypatch.setitem(
        sys.modules,
        "stereo_depth",
        SimpleNamespace(
            load_rig=lambda *_: SimpleNamespace(left="left", right="right"),
            stereo_depth=lambda *_: (np.ones((20, 20)), np.ones((20, 20), bool)),
        ),
    )
    observations = None
    if with_depth:
        import hashlib

        (tmp_path / "meta").mkdir()
        calibration = tmp_path / "meta/camera_calibration.json"
        calibration.write_text("test calibration bytes")
        frames = []
        for frame in range(2):
            np.save(tmp_path / f"depth-{frame}.npy", np.ones((20, 20)))
            cv2.imwrite(str(tmp_path / f"valid-{frame}.png"), target)
            frames.append(
                {
                    "id": frame,
                    "depth_path": f"depth-{frame}.npy",
                    "depth_validity": f"valid-{frame}.png",
                }
            )
        observations = tmp_path / "observations.json"
        observations.write_text(
            json.dumps(
                {
                    "episode": 1,
                    "units": "metres",
                    "depth": {"encoding": "metres"},
                    "frames": frames,
                    "color_camera": {"id": "ego_cam_c", "fx": 100, "fy": 100},
                    "input_contract": {
                        "reference_pose_tables_read": False,
                        "calibration_sha256": hashlib.sha256(
                            calibration.read_bytes()
                        ).hexdigest(),
                    },
                }
            )
        )
    result = select(
        tmp_path, masks, None, 1, "target", tmp_path / "selection.json", observations
    )
    assert result["selected"]["frame"] == 1
    assert result["excluded_unknown_other_foreground_frames"] == [0]
    assert set(result["selected_foreground_sha256"]) == {"target", "other"}
    if with_depth:
        assert result["area_measure"] == "metric_area_proxy_m2"
        assert result["selected"]["measured_foreground_pixels"] == 400
        record = json.loads(observations.read_text())
        record["input_contract"]["reference_pose_tables_read"] = True
        observations.write_text(json.dumps(record))
        with pytest.raises(ValueError, match="label-free"):
            select(
                tmp_path, masks, None, 1, "target", tmp_path / "bad.json", observations
            )


def test_isolated_object_is_preferred_to_a_larger_fused_foreground():
    empty = np.zeros((100, 100), bool)
    one = empty.copy()
    one[10:60, 10:40] = True
    stacked = one.copy()
    stacked[60:90, 10:40] = True
    other = empty.copy()
    other[60:90, 10:40] = True
    rows = [
        dict(frame=155, **mask_features(stacked, other, empty)),
        dict(frame=0, **mask_features(one, other, empty)),
    ]
    ranked, _ = rank(rows)
    assert ranked[0]["frame"] == 0


def test_hand_occlusion_and_tiny_foreground_do_not_win_frame_selection():
    empty = np.zeros((100, 100), bool)
    target = empty.copy()
    target[10:90, 10:90] = True
    hand = empty.copy()
    hand[50:90, 10:90] = True
    tiny = empty.copy()
    tiny[10:12, 10:12] = True
    rows = [
        dict(frame=0, **mask_features(target, empty, hand)),
        dict(frame=20, **mask_features(target, empty, empty)),
        dict(frame=1, **mask_features(tiny, empty, empty)),
    ]
    ranked, _ = rank(rows)
    assert ranked[0]["frame"] == 20 and not any(r["frame"] == 1 for r in ranked)
    with pytest.raises(ValueError):
        mask_features(target, np.zeros((2, 2)), empty)
