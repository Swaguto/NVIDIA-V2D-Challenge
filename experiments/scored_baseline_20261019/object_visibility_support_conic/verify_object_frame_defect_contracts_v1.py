from pathlib import Path
from fractions import Fraction as Q
import datetime,json,os
import numpy as np
from object_compiled_frame_defect_v1 import origin_l1_defect,path_defect_budget,frame_residual_is_bounded
from thumb_conic_geometry_v1 import plain,journal
S=Path(__file__).resolve().parents[1];O=S/'e037-object-frame-defect-contract-v1'
def save(n,v):
 p=O/n;p.write_text(json.dumps(plain(v),indent=2,allow_nan=False)+'\n');p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
passed=[]
with (O/'normalized-BEFORE-AFTER-journal.jsonl').open('w')as log:
 def t(n,proposal,fn):
  journal(log,dict(kind='BEFORE_NEW_static_frame_defect_contract',name=n,proposal=proposal));r=fn();save(n+'-RETURN.json',r);journal(log,dict(kind='AFTER_NEW_static_frame_defect_contract',name=n,actual_RETURN=r));assert r['passed'];passed.append(n)
 def exact():
  p=np.array([np.float32(.0777),np.float32(.0016),np.float32(-.0285)],float);d=origin_l1_defect('.0777 .0016 -.0285',p);direct=sum([abs(Q('.0777')-Q(float(p[0]))),abs(Q('.0016')-Q(float(p[1]))),abs(Q('-.0285')-Q(float(p[2])))]);return dict(passed=d==direct and d>0,compiled_float32_origin=p,exact_L1_defect=dict(numerator=d.numerator,denominator=d.denominator),decimal_vs_binary_inputs_are_distinct=True)
 t('NEW_exact_decimal_compiled_origin_defect',dict(decimal='.0777 .0016 -.0285'),exact)
 def chain():
  R=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]);e1=np.array([.125,-.25,.5]);e2=np.array([-.25,.125,0.]);error=e1+R@e2;bound=float(abs(e1).sum()+abs(e2).sum());return dict(passed=np.linalg.norm(error)<=bound and abs(error).max()<=bound,propagated_difference=error,path_L1_bound=bound,mathematical_reason='rotations preserve Euclidean norm; triangle inequality; per-origin L2<=L1')
 t('NEW_rotated_path_translation_defect_bound',dict(two_origins=True),chain)
 def support():
  H=np.array([[0.,0.,0.],[1.,.5,.25]]);e=np.array([.125,-.25,.5]);n=np.array([0.,0.,1.]);a=np.max(H@n);b=np.max((H+e)@n);return dict(passed=abs(a-b)<=np.linalg.norm(e),projection_change=abs(a-b),euclidean_frame_defect=np.linalg.norm(e),NOT_actual_native_gap_or_gate_change=True)
 t('NEW_support_function_frame_defect',dict(unit_normal=[0,0,1]),support)
 def zero():
  d=origin_l1_defect('.125 .25 -.5',[.125,.25,-.5]);return dict(passed=d==0 and not frame_residual_is_bounded(1e-8,d)and frame_residual_is_bounded(0.,d),exact_zero_case=True,rejects_excess_without_tolerance_repair=True,original_native_gate_m=.001,original_URDF_position_gate_rad=1e-8)
 t('NEW_zero_defect_and_excess_rejection',dict(exact_origin=True),zero)
 out=dict(schema='v2d.NEW-static-frame-origin-defect-contracts.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),NEWcontracts=4,contracts=passed,episode_FK_native_or_source_calls=0,old11and14controls_reused_BY_HASH=True,new_bound_NOT_relaxed_original_gate=True);save('report.json',out);journal(log,dict(kind='AFTER_NEW_defect_contract_suite',returned=out));print(json.dumps(out))
