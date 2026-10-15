"""Compare fresh IK with frozen targets without relaxing the declared tolerance."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

def compare(reference,candidate):
 with np.load(reference/'retarget.npz',allow_pickle=False) as expected,np.load(candidate/'retarget.npz',allow_pickle=False) as got:
  for k in ['frame_indices','robot_valid']:np.testing.assert_array_equal(expected[k],got[k],err_msg=k)
  mask=expected['robot_valid'];np.testing.assert_array_equal(np.isnan(expected['robot_qpos']),np.isnan(got['robot_qpos']))
  if not np.isfinite(got['robot_qpos'][mask]).all():raise ValueError('Nonfinite supported robot state')
  delta=np.abs(expected['robot_qpos'][mask]-got['robot_qpos'][mask])
  maxima=[float(delta[:,:3].max()),float(delta[:,3:7].max()),float(delta[:,7:].max())] if mask.any() else [0.,0.,0.]
 q=json.loads((candidate/'quality.json').read_text())
 limits=not q['joint_position_limit_violations'] and not any(v['finger_velocity_limit_violating_pairs'] for v in q['per_hand'].values())
 strict=max(maxima)<=1e-6 and limits
 report={'status':'passed' if strict else 'failed','atol_qpos':1e-6,'max_absolute_qpos_difference':max(maxima),'maximum_root_translation_difference_m':maxima[0],'maximum_quaternion_component_difference':maxima[1],'maximum_finger_angle_difference_rad':maxima[2],'frame_axes_and_unknown_masks_match':True,'joint_limits_verified':limits,'physical_grasp_success_tested':False,'scope':'Strict numerical target comparison. A failure is retained; use frozen targets for exact target playback.'}
 (candidate/'checkpoint-comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));return report
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);a=p.parse_args();r=compare(a.reference,a.candidate)
 if r['status']!='passed':raise SystemExit(1)
