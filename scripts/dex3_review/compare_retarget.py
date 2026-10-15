"""Compare a fresh IK result with the frozen checkpoint, retaining unknown masks."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

def compare(reference,candidate):
 with np.load(reference/'retarget.npz',allow_pickle=False) as expected,np.load(candidate/'retarget.npz',allow_pickle=False) as got:
  for k in ['frame_indices','robot_valid']:np.testing.assert_array_equal(expected[k],got[k],err_msg=k)
  mask=expected['robot_valid'];np.testing.assert_array_equal(np.isnan(expected['robot_qpos']),np.isnan(got['robot_qpos']))
  # Fixed before rerun: one micrometre for root translation; one microradian
  # / quaternion-component unit for the remaining configuration coordinates.
  np.testing.assert_allclose(expected['robot_qpos'][mask],got['robot_qpos'][mask],rtol=0,atol=1e-6)
  maximum=float(np.max(np.abs(expected['robot_qpos'][mask]-got['robot_qpos'][mask]))) if mask.any() else 0.
 q=json.loads((candidate/'quality.json').read_text())
 if q['joint_position_limit_violations'] or any(v['finger_velocity_limit_violating_pairs'] for v in q['per_hand'].values()):raise ValueError('Fresh IK violates joint bounds')
 report={'status':'passed','atol_qpos':1e-6,'max_absolute_qpos_difference':maximum,'frame_axes_and_unknown_masks_match':True,'joint_limits_verified':True,'physical_grasp_success_tested':False}
 (candidate/'checkpoint-comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));return report
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);a=p.parse_args();compare(a.reference,a.candidate)
