"""The Docker entrypoint must forward the same safety bounds as native runs."""
import runpy
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[3]


def test_docker_cli_forwards_pose_limits(monkeypatch):
    captured = {}
    container = ModuleType('v2d.docker.container')
    container.run_in_container = lambda **kwargs: captured.update(kwargs)
    config = ModuleType('v2d.gsplat_refinement.docker._config')
    config.IMAGE_NAME = 'test-image'
    config.MODULES_DIR = 'test-modules'
    for name, module in ((container.__name__, container), (config.__name__, config)):
        monkeypatch.setitem(sys.modules, name, module)
    limits = {
        'max_object_translation_change': 0.003,
        'max_object_rotation_change_deg': 0.5,
        'max_hand_translation_change': 0.002,
        'max_hand_rotation_change_deg': 0.5,
        'max_hand_articulation_change_deg': 1.0,
        'max_camera_translation_change': 0.002,
        'max_camera_rotation_change_deg': 0.25,
    }
    argv = ['run_refine_simple', '--frames_dir', '/frames', '--depth_dir', '/depth',
            '--intrinsics_path', '/intrinsics.json', '--overlay_path', '/out/overlay.mp4']
    for name, value in limits.items():
        argv.extend(['--' + name, str(value)])
    monkeypatch.setattr(sys, 'argv', argv)
    runpy.run_path(str(ROOT / 'reconstruction/modules/v2d_gsplat_refinement/docker/run_refine_simple.py'),
                   run_name='__main__')
    assert captured['module'] == 'v2d.gsplat_refinement.lib.refine_simple'
    assert captured['gpus'] is True
    assert {key: captured['extra_args'][key] for key in limits} == limits
