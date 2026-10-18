"""A matching partial trace must not become a reproducibility certificate."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("compare_startup", ROOT/"compare_startup_traces.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
spec = importlib.util.spec_from_file_location("trace_startup", ROOT/"v2d_dex3_flashchord/startup_trace.py")
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)


class ReadOnly:
    def __init__(self, value):self.value = np.asarray(value)
    def numpy(self):return self.value
    def assign(self, *_):raise AssertionError("Trace writes physics state")
    def zero_(self):raise AssertionError("Trace clears physics state")


def env(complete=True):
    a = lambda:ReadOnly(np.array([[1.,2.,3.]], np.float32))
    if not complete:return SimpleNamespace(state_0=SimpleNamespace(joint_q=a()))
    return SimpleNamespace(
        state_0=SimpleNamespace(**{k:a() for k in ["joint_q","joint_qd","body_q","body_qd"]}),
        solver=SimpleNamespace(mjw_data=SimpleNamespace(**{k:a() for k in ["qpos","qvel","qacc_warmstart","time","act","xfrc_applied"]})),
        timestep=a(), episode_step=a(), observation=a(), action=SimpleNamespace(joint_target=a()),
        control=SimpleNamespace(joint_target_pos=a(),joint_f=a()))


def save(path, state, change=None, stages=m.REQUIRED_STAGES):
    path.mkdir()
    trace = t.StartupTrace()
    for stage in stages:
        if change:change(stage,state)
        trace.capture(stage,state,pending_action=np.ones((1,2)) if stage=="before-first-control-interval" else None)
    trace.save(path)
    return path


def test_complete_and_partial_equal_traces_have_different_coverage(tmp_path):
    a=save(tmp_path/'a',env());b=save(tmp_path/'b',env());r=m.compare(a,b)
    assert r['required_startup_buffers_finite_and_bit_exact']
    assert not r['physics_executed'] and not r['replay_variability_cause_proven']
    c=save(tmp_path/'c',env(False));d=save(tmp_path/'d',env(False));r=m.compare(c,d)
    assert r['all_recorded_fields_bit_exact']
    assert not r['required_startup_buffers_finite_and_bit_exact']
    assert any(x.get('field')=='native.qacc_warmstart' for x in r['coverage_failures'])


def test_first_difference_is_at_the_recorded_reset_boundary(tmp_path):
    a=save(tmp_path/'a',env())
    def change(stage,state):
        if stage=='after-explicit-reset':state.solver.mjw_data.qacc_warmstart.value[:]=7
    b=save(tmp_path/'b',env(),change);r=m.compare(a,b)
    assert r['first_recorded_difference']==dict(index=3,stage='after-explicit-reset',field='native.qacc_warmstart',kind='recorded_buffer_difference')
    assert not r['required_startup_buffers_finite_and_bit_exact']
    assert not r['replay_variability_cause_proven']


def test_tampered_field_is_rejected_even_with_updated_archive_hash(tmp_path):
    a=save(tmp_path/'a',env());p=a/'startup-state-trace.json';r=json.loads(p.read_text())
    key=r['snapshots'][0]['fields']['native.qpos']['key']
    with np.load(a/'startup-state-trace.npz') as z:arrays={k:z[k].copy() for k in z.files}
    arrays[key][:]=9
    np.savez_compressed(a/'startup-state-trace.npz',**arrays)
    r['arrays_sha256']=m.digest(a/'startup-state-trace.npz');p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='field hash'):m.load_trace(a)


def test_matching_nonfinite_state_cannot_qualify_startup(tmp_path):
    e=env();e.solver.mjw_data.qpos.value[:]=np.nan
    a=save(tmp_path/'a',e);b=save(tmp_path/'b',e);r=m.compare(a,b)
    assert r['all_recorded_fields_bit_exact']
    assert not r['required_startup_buffers_finite_and_bit_exact']
    assert any(x['reason']=='nonfinite_required_buffer' for x in r['coverage_failures'])


def test_mismatched_lifecycle_cannot_be_aligned_by_array_similarity(tmp_path):
    a=save(tmp_path/'a',env());b=save(tmp_path/'b',env(),stages=m.REQUIRED_STAGES[::-1]);r=m.compare(a,b)
    assert not r['lifecycle_order_equal']
    assert not r['required_startup_buffers_finite_and_bit_exact']


def test_unsigned_delta_does_not_wrap_and_readback_size_is_bounded(tmp_path):
    e=env();e.world_count=np.array([0],np.uint8);a=save(tmp_path/'a',e)
    e.world_count[:]=255;b=save(tmp_path/'b',e);r=m.compare(a,b)
    field=next(x for x in r['comparisons'][0]['fields'] if x['name']=='env.world_count')
    assert field['maximum_finite_absolute_difference']==255
    with pytest.raises(ValueError,match='readback limit'):m.load_trace(a,max_uncompressed_bytes=1)


def test_extended_native_persistent_inputs_are_passive_copies(tmp_path):
    e=env();e.action.raw_action=ReadOnly([123.]);e.action.buffers=SimpleNamespace(raw_action=ReadOnly([321.]));before=e.solver.mjw_data.act.value.copy();a=save(tmp_path/'a',e)
    with np.load(a/'startup-state-trace.npz') as z:
        np.testing.assert_array_equal(z['snapshot_000__native__act'],before)
        np.testing.assert_array_equal(z['snapshot_000__native__xfrc_applied'],e.solver.mjw_data.xfrc_applied.value)
        np.testing.assert_array_equal(z['snapshot_000__action__raw_action'],[123.])
        np.testing.assert_array_equal(z['snapshot_000__action__buffers__raw_action'],[321.])
    np.testing.assert_array_equal(e.solver.mjw_data.act.value,before)
    m.load_trace(a)


def test_active_trace_is_not_accepted_as_passive_evidence(tmp_path):
    a=save(tmp_path/'a',env());p=a/'startup-state-trace.json';r=json.loads(p.read_text())
    r['simulation_buffers_written']=True;p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='passive trace'):m.load_trace(a)


def test_changed_trace_producer_is_an_explicit_coverage_failure(tmp_path):
    a=save(tmp_path/'a',env());b=save(tmp_path/'b',env());p=b/'startup-state-trace.json';r=json.loads(p.read_text())
    r['producer_sha256']='0'*64;p.write_text(json.dumps(r));result=m.compare(a,b)
    assert result['all_recorded_fields_bit_exact']
    assert not result['required_startup_buffers_finite_and_bit_exact']
    assert not result['all_model_and_solver_state_recorded']
