"""Encode the actual worker source overlay with odd and even calibration grids."""

import json
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageFont

from issue6.worker import render_source_camera


@pytest.mark.parametrize("width,height", [(17, 11), (18, 12)])
def test_source_overlay_encodes_without_changing_input_grid(tmp_path, monkeypatch, width, height):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe are required; installed in Linux CI")

    class Scene:
        def __init__(self, **kwargs):
            pass

        def add(self, *args, **kwargs):
            pass

    monkeypatch.setitem(sys.modules, "pyrender", SimpleNamespace(Scene=Scene))
    renderer = SimpleNamespace(render=lambda scene: (np.zeros((height, width, 3), np.uint8), np.zeros((height, width))))
    render = SimpleNamespace(_add_lights=lambda *args: None, _font=lambda size: ImageFont.load_default())
    folder = tmp_path / "align/frames"
    folder.mkdir(parents=True)
    frames = [{"id": 10, "timestamp_s": 0}, {"id": 11, "timestamp_s": 0.05}]
    for frame in frames:
        Image.new("RGB", (width, height)).save(folder / f"{frame['id']:06d}.png")
    render_source_camera(
        tmp_path, {"frames": frames, "camera": {"width": width, "height": height}}, {}, [], renderer, None, render
    )
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames",
            "-of",
            "json",
            str(folder.parent / "hand_mesh_overlay.mp4"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert (stream["width"], stream["height"]) == (18, 12)
    assert int(stream["nb_frames"]) == 2
    for frame in frames:
        with Image.open(folder / f"{frame['id']:06d}.png") as source:
            assert source.size == (width, height)
