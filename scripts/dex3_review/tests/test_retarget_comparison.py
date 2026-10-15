"""A changed inferred pose must not be reported as exact reproduction."""
import importlib.util,json
from pathlib import Path
import numpy as np
import pytest
spec=importlib.util.spec_from_file_location('compare_retarget',Path(__file__).resolve().parents[1]/'compare_retarget.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def inputs(tmp_path):
 mask=np.array([[True,False],[True,True]]);q=np.zeros((2,2,14));q[:,:,6]=1;q[~mask]=np.nan
 roots=[tmp_path/'reference',tmp_path/'candidate']
 for root in roots:
  root.mkdir();np.savez(root/'retarget.npz',frame_indices=np.array([10,11]),robot_valid=mask,robot_qpos=q)
 (roots[1]/'quality.json').write_text(json.dumps({'joint_position_limit_violations':0,'per_hand':{'left':{'finger_velocity_limit_violating_pairs':0},'right':{'finger_velocity_limit_violating_pairs':0}}}))
 return roots,mask,q

def test_identical_targets_pass(tmp_path):
 (a,b),mask,q=inputs(tmp_path);assert m.compare(a,b)['status']=='passed'

def test_small_finger_change_remains_a_failed_strict_comparison(tmp_path):
 (a,b),mask,q=inputs(tmp_path);q[0,0,7]=1e-5;np.savez(b/'retarget.npz',frame_indices=np.array([10,11]),robot_valid=mask,robot_qpos=q)
 r=m.compare(a,b);assert r['status']=='failed';assert r['atol_qpos']==1e-6
 assert json.loads((b/'checkpoint-comparison.json').read_text())['status']=='failed'

def test_filling_unknown_geometry_is_rejected(tmp_path):
 (a,b),mask,q=inputs(tmp_path);q[0,1]=0;np.savez(b/'retarget.npz',frame_indices=np.array([10,11]),robot_valid=mask,robot_qpos=q)
 with pytest.raises(AssertionError):m.compare(a,b)
 assert not (b/'checkpoint-comparison.json').exists()
