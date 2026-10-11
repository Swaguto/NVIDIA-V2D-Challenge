from PIL import Image

from issue6.backend import prepare_render_inputs
from issue6.manifest import read_json, write_json


def test_final_overlay_omits_unsupported_mesh_without_removing_raw_record(tmp_path):
    frames = tmp_path / "frames"
    records = tmp_path / "raw/3"
    frames.mkdir()
    axis = [{"id": 191, "timestamp_s": 9.43}, {"id": 192, "timestamp_s": 9.48}]
    for frame in axis:
        Image.new("RGB", (2, 2)).save(frames / f"{frame['id']:06d}.png")
        write_json(
            records / f"{frame['id']:06d}.json",
            {"frame_idx": frame["id"], "diagnostics": {"metric_valid": frame["id"] == 191}},
        )
    _, supported = prepare_render_inputs(frames, records.parent, axis, tmp_path / "display", metric_supported_only=True)
    assert read_json(supported / "3/000000.json")["original_frame_idx"] == 191
    assert not (supported / "3/000001.json").exists()
    assert read_json(records / "000192.json")["diagnostics"]["metric_valid"] is False
