"""Exercise participant-kit score() wrappers, with explicitly synthetic provenance."""
import base64
import importlib.util
import io
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/eval"))
from official_metric import score_episode
from glb_mesh import load_vertices

KIT = Path(os.environ.get("V2D_SUBMISSION_KIT",
    ROOT.parent / "runs/issue5_diagnostic_20261003/kit/v2d_submission_kit")) / "metric_code/track_3"


@pytest.fixture(params=['AUC.py', 'SP-SR.py', 'MP-SR.py', 'RPE.py', 'MPPE.py'])
def official(request):
    path = KIT / request.param
    if not path.exists(): pytest.skip('participant kit is required')
    spec = importlib.util.spec_from_file_location('official_' + request.param.replace('-', '_'), path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def frames(module, candidate, reference, vertices):
    solution, submission = [], []
    # Fixture hashes and commit URL satisfy syntax only. They describe no rollout.
    provenance = {key: 'synthetic-test-only' for key in module._PROVENANCE_COLUMNS}
    provenance['code_commit_url'] = 'https://github.com/test/synthetic-fixture/commit/' + '0' * 40
    for t in range(candidate.shape[0]):
        for world in range(candidate.shape[1]):
            for body in range(candidate.shape[2]):
                row_id = f'synthetic/{t}/{world}/{body}'
                solution.append({'row_id': row_id, 'episode_index': 0, 'world_index': world,
                    'frame_index': t, 'object_slot': body, 'object_name': f'body_{body}',
                    **dict(zip(module._REF_COLUMNS, reference[t, body]))})
                submission.append({'row_id': row_id, **dict(zip(module._POSE_COLUMNS, candidate[t, world, body])),
                                   **provenance})
    solution, submission = pd.DataFrame(solution), pd.DataFrame(submission)
    buffer = io.BytesIO(); np.savez(buffer, **{f'body_{b}': v for b, v in enumerate(vertices)})
    solution['auc_vertex_samples_b64'] = None
    solution.loc[0, 'auc_vertex_samples_b64'] = base64.b64encode(buffer.getvalue()).decode()
    return solution, submission


@pytest.mark.parametrize('control', ['oracle', 'moving', 'scale', 'multiworld'])
def test_full_wrapper_matches_adapter(official, control):
    ref = np.zeros((6, 2, 7)); ref[..., 6] = 1
    ref[:, 0, :3] = [.1, .2, .7]; ref[:, 1, :3] = [.2, .2, .7]
    candidate = ref[:, None].copy()
    if control == 'moving': candidate[:, :, :, 0] += np.arange(6)[:, None, None] * .03
    if control == 'scale': candidate[..., :3] *= 1.2
    if control == 'multiworld':
        candidate = np.repeat(candidate, 3, axis=1)
        candidate[:, 1, :, :3] += [1, -.5, .2]
        candidate[:, 2, :, :3] += [-.4, 1, .1]
    vertices = np.random.default_rng(0).normal(scale=.05, size=(2, 32, 3))
    solution, submission = frames(official, candidate, ref, vertices)
    expected = score_episode(candidate, ref, vertices)[official._METRIC]
    assert official.score(solution, submission, 'row_id') == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize('defect', ['missing_row', 'duplicate_row', 'bad_provenance', 'bad_commit', 'nonfinite', 'zero_quaternion'])
def test_full_wrapper_rejects_invalid_rows_and_provenance(official, defect):
    ref = np.zeros((3, 2, 7)); ref[..., 6] = 1
    vertices = np.random.default_rng(1).normal(size=(2, 16, 3))
    solution, submission = frames(official, ref[:, None], ref, vertices)
    if defect == 'missing_row': submission = submission.iloc[1:].copy()
    if defect == 'duplicate_row': submission = pd.concat([submission, submission.iloc[:1]])
    if defect == 'bad_provenance': submission.loc[0, 'eval_code_sha256'] = 'different-build'
    if defect == 'bad_commit': submission['code_commit_url'] = 'https://github.com/test/synthetic-fixture/tree/main'
    if defect == 'nonfinite': submission.loc[0, 'pos_x'] = np.inf
    if defect == 'zero_quaternion': submission.loc[0, ['quat_x', 'quat_y', 'quat_z', 'quat_w']] = 0
    with pytest.raises(official.ParticipantVisibleError): official.score(solution, submission, 'row_id')


def test_evaluator_glb_reader_matches_full_scene_and_nodes(tmp_path):
    import trimesh
    scene = trimesh.Scene()
    transform = np.eye(4); transform[:3, 3] = [2, 0, 0]
    scene.add_geometry(trimesh.creation.box(), geom_name='box', node_name='instance_a', transform=transform)
    transform = np.eye(4); transform[:3, :3] *= 2; transform[:3, 3] = [-2, 0, 0]
    scene.add_geometry(trimesh.creation.icosphere(subdivisions=1), geom_name='sphere', node_name='instance_b', transform=transform)
    path = tmp_path / 'scene.glb'; scene.export(path)
    with path.open('rb') as file: vertices, faces = load_vertices(file)
    full = trimesh.load(path).to_mesh()
    np.testing.assert_allclose(np.stack([vertices.min(0), vertices.max(0)]), full.bounds)
    assert len(vertices) == len(full.vertices)
    assert len(faces) == full.faces.size
