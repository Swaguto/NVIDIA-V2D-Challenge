"""NEW bounds and controls; all source data/old proof values are reused by hash."""
from pathlib import Path
from fractions import Fraction as Q
import ctypes,datetime,hashlib,itertools,json,math,os
import numpy as np,trimesh
from training_proxy_roundoff_v1 import U,ETA,G5,T5,fraction,pack,unpack,outward,pipeline_bound,proxy_allowance
from thumb_conic_geometry_v1 import plain,journal
S=Path(__file__).resolve().parents[1];O=S/'e037-training-proxy-roundoff-v1'
T=S/'e037-training-source-constraints-v1';M=S/'e037-training-orderstat-support-v1';D=S/'e037-object-dynamic-runtime-v1'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def exact(v):
 if isinstance(v,Q):return pack(v)
 if isinstance(v,dict):return {k:exact(x)for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [exact(x)for x in v]
 return v
def save(n,v):
 p=O/n;p.write_text(json.dumps(plain(exact(v)),indent=2,allow_nan=False)+'\n');p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return dict(path=str(p),sha256=sha(p))
with(O/'normalized-BEFORE-AFTER-journal.jsonl').open('w')as log:
 def emit(v):journal(log,exact(v))
 scope=dict(schema='v2d.NEW-training-proxy-roundoff-declaration.v1',
  mathematical_scope='binary64 round-nearest, finite operations, no overflow; arbitrary3term dot order <=5rounded operations (FMA covered by this conservative bound)',
  uniform_world_translation_halfwidth_m=.02,box_NOT_original_object_or_robot_gate=True,
  source_operator_scope='training helper camera-translation update and point-local transform; KD-tree distances/raster/original source truth NOT certified by these bounds',
  proposal_guard_OUTWARD_only_actual_hard_training_and_original_source_gates_STILL_REQUIRED=True,
  no_source_metric_NN_old_controls_or_old124defect_audit_rerun=True,
  original_radii_arrays_124defects_and_candidate_tolerances_UNCHANGED=True,
  no_actual_nonzero_episode_point_or_solver_selected=True)
 save('ONE-declared-uniform-bound-scope.json',scope);emit(dict(kind='BEFORE_NEW_uniform_error_contracts',scope=scope))
 lib=ctypes.CDLL(None);lib.fegetround.restype=ctypes.c_int;rounding=int(lib.fegetround())
 save('actual-read-only-rounding-mode-RETURN.json',dict(fegetround_RETURN=rounding,mode_NOT_changed=True))
 assert rounding==0,'UNRESOLVED non-nearest rounding mode'
 controls=[]
 def control(n,v,ok):
  save(n+'-RETURN.json',v);emit(dict(kind='AFTER_NEW_arithmetic_contract_RETURN',name=n,actual_RETURN=v));assert ok,n;controls.append(n)
 # NEW three-term cancellation/order envelopes, unlike old binary32 objectives.
 a=[2.**40,-2.**40,2.**-40];c=[1.,1.,1.];truth=sum(Q(x)*Q(y)for x,y in zip(a,c));returns=[]
 bound=G5*sum(abs(Q(x)*Q(y))for x,y in zip(a,c))+T5
 for order in itertools.permutations(range(3)):
  products=[float(a[i]*c[i])for i in order];value=float(float(products[0]+products[1])+products[2]);returns.append(dict(order=order,value=value,error=Q(value)-truth))
 control('NEW-dot-cancellation-all-associations',dict(exact_value=truth,bound=bound,returns=returns),all(abs(v['error'])<=bound for v in returns))
 # NEW entire rounded translation-to-local pipeline at distinct synthetic boxes.
 R=np.array([[1.,.125,0],[-.0625,1.,.25],[.03125,0,1.]]);RC=np.array([[1.,.125,0],[0,1.,-.0625],[.125,0,1.]]);P=np.array([.75,-.25,.5]);t=np.array([.125,-.125,.25]);V=np.array([.125,.0625,-.03125]);H=Q(1,64);b=pipeline_bound(list(map(lambda x:Q(abs(x)),P)),list(map(Q,t)),[[Q(x)for x in row]for row in R],[[Q(x)for x in row]for row in RC],list(map(lambda x:Q(abs(x)),V)),H);rets=[]
 for signs in itertools.product([-1,1],repeat=3):
  delta=np.array([float(H*s)for s in signs]);updated=t+np.einsum('ij,j->i',RC,delta,optimize=False);actual=np.einsum('i,ij->j',P-updated,R,optimize=False)-V
  ideal=[sum((Q(P[j])-Q(t[j])-sum(Q(RC[j,k])*Q(delta[k])for k in range(3)))*Q(R[j,i])for j in range(3))-Q(V[i])for i in range(3)];errors=[Q(actual[i])-ideal[i]for i in range(3)];rets.append(dict(delta=delta,actual=actual,exact_error=errors))
 control('NEW-full-source-transform-forward-bound',dict(bound=b,returns=rets),all(abs(e)<=b['rounded_offset_error'][i]for v in rets for i,e in enumerate(v['exact_error'])))
 # Outward conversion must not round away small guards. This does NOT change old radii.
 cases=[]
 for r,e in [(Q(1,8),Q(1,2**70)),(Q(0),Q(1,2**1075)),(Q(1,2**500),Q(1,2**600))]:
  x=outward(r+e);cases.append(dict(radius=r,allowance=e,outward=x,excess=Q(x)-r-e))
 control('NEW-outward-radius-rounding-and-subnormal',dict(cases=cases),all(v['excess']>=0 for v in cases))
 # A componentwise bound gives a Euclidean bound via L1, with signs retained.
 E=[Q(1,8),Q(1,16),Q(1,32)];signs=list(itertools.product([-1,1],repeat=3));norms=[sum((s*e)**2 for s,e in zip(ss,E))for ss in signs];l1=sum(E)
 control('NEW-vector-to-distance-error-certificate',dict(component=E,squared_errors=norms,L1=l1),all(v<=l1*l1 for v in norms))
 # Positive radius converts inherited SQUARED defects; zero radius stays UNKNOWN.
 cases=[]
 for r,d in [(Q(1,8),Q(1,2**80)),(Q(0),Q(1,2**80))]:
  cases.append(dict(radius=r,defect=d,status='UNRESOLVED_zero_radius'if not r else'positive_radius_rational_bound',norm_excess_upper=None if not r else d/(2*r)))
 control('NEW-positive-radius-defect-transfer-and-zero-unknown',dict(cases=cases),cases[0]['norm_excess_upper']>0 and cases[1]['norm_excess_upper']is None)
 assert len(controls)==5
 save('all5-NEW-roundoff-contracts.json',dict(contracts=controls,old26training_orderstat_controls_reused_BY_HASH=True))
 Fdir=S/'e037-thumb-ray-independent-validation-v1/frozen';manifest=read(Fdir/'manifest.json');mesh=trimesh.load(Fdir/manifest['objects'][0]['mesh'],force='mesh',process=False);Vmax=[Q(float(x))for x in np.max(np.abs(np.asarray(mesh.vertices,float)),axis=0)]
 with np.load(D/'dynamic-source-data.npz')as z:poses=z['original_camera_from_object'];Rcw=z['camera_from_world_linear']
 records=[];all_covered=True;max_allow=Q(0)
 for frame in range(151):
  emit(dict(kind='BEFORE_NEW_uniform_frame_bound',frame=frame,unassigned_continuous_delta=True))
  point=T/f'frame-{frame:03d}-NEW-training-point-RETURN.npz';mp=M/f'frame-{frame:03d}-NEW-proxy-SOC-maps-RETURN.npz';errfile=M/f'frame-{frame:03d}-NEW-exact-affine-map-errors.json';binding=read(M/f'frame-{frame:03d}-NEW-independent-proxy-binding.json')
  with np.load(point)as z:Pmax=[Q(float(x))for x in np.max(np.abs(z['points_C_m']),axis=0)]
  with np.load(mp)as z:radii=z['nominal_radius_m'];rmin=Q(float(radii.min()));oldA=z['canonical_A'];oldb=z['canonical_b'];lower=float(radii[0]);upper=float(radii[-1])
  errors=[[Q(0)for _ in range(3)]for _ in range(3)]
  for row in read(errfile):errors[row['row']][row['column']]=unpack(row['exact_error'])
  R=[[Q(float(x))for x in row]for row in poses[frame,:3,:3]];RC=[[Q(float(x))for x in row]for row in Rcw[frame]];t0=[Q(float(x))for x in poses[frame,:3,3]]
  result=proxy_allowance(Pmax,t0,R,RC,Vmax,Q(.02),errors);allow=result['L1_distance_allowance'];dmax=unpack(binding['maximum_positive_exact_squared_defect']);needed=None if not rmin else dmax/(2*rmin);covered=needed is not None and needed<=allow
  guarded_lower=outward(Q(lower)+allow);guarded_upper=outward(Q(upper)+allow)
  ret=dict(frame=frame,original_map_file=str(mp),original_map_sha256=sha(mp),old_affine_errors_file=str(errfile),old_affine_errors_sha256=sha(errfile),cached_point_file=str(point),cached_point_file_sha256=sha(point),original_radii_UNALTERED=True,
   P_component_absolute=Pmax,V_component_absolute=Vmax,T_translation=t0,R_binary=R,RC_binary=RC,world_halfwidth=Q(.02),inherited_exact_map_errors=errors,
   arithmetic_bounds=result,original_lower_radius=lower,original_upper_radius=upper,minimum_original_radius=rmin,
   inherited_max_positive_squared_defect=dmax,old_defect_to_norm_upper=needed,allowance_covers_inherited_nominal_defects=covered,
   NEW_outward_proposal_lower_radius=guarded_lower,NEW_outward_proposal_upper_radius=guarded_upper,
   NEW_lower_radius_addition_exact_rounding_excess=Q(guarded_lower)-Q(lower)-allow,NEW_upper_radius_addition_exact_rounding_excess=Q(guarded_upper)-Q(upper)-allow,
   no_KD_or_raster_or_original_source_gate_certification=True,actual_hard_training_oracle_and_original_gates_REQUIRED=True)
  save(f'frame-{frame:03d}-NEW-uniform-proxy-error-RETURN.json',ret);emit(dict(kind='AFTER_NEW_uniform_frame_bound_RETURN',actual_RETURN=ret));assert allow>=0
  records.append(ret);all_covered=all_covered and covered;max_allow=max(max_allow,allow)
 save('all151-NEW-uniform-error-bindings.json',records)
 report=dict(schema='v2d.NEW-training-proxy-roundoff.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),NEW_analytic_contracts=5,frames=151,maximum_uniform_L1_distance_allowance_m=pack(max_allow),maximum_uniform_L1_distance_allowance_m_float=outward(max_allow),all_inherited_nominal_F_defects_covered=all_covered,old124defects_and_radii_UNALTERED=True,guarded_proposal_model_NOT_actual_source_feasibility=True,KD_raster_source_accuracy_uncertainty_remains=True,no_actual_nonzero_episode_point_or_solver=True,no_old_source_NN_controls_or124defect_audit_replayed=True,all_original_tolerances_and_gates_UNCHANGED=True)
 save('report.json',report);emit(dict(kind='AFTER_NEW_uniform_bound_report_RETURN',actual_RETURN=report));print(json.dumps(report))
