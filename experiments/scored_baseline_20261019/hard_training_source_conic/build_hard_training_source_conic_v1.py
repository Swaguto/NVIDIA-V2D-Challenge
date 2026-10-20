"""NEW median-conic object model plus independent forward-error proof.

No solver or source operator is called. The conic model is a proposal generator;
actual hard-training and ORIGINAL source/motion/native gates remain separate.
"""
from pathlib import Path
from fractions import Fraction as Q
import datetime,hashlib,itertools,json,os
import numpy as np
from scipy import sparse
from thumb_conic_geometry_v1 import plain,journal

S=Path(__file__).resolve().parents[1];O=S/'e037-hard-training-source-model-v1'
B=S/'e037-training-proxy-roundoff-v1';M=S/'e037-training-orderstat-support-v1';D=S/'e037-object-dynamic-runtime-v1'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
unpack=lambda d:Q(int(d['numerator']),int(d['denominator']))
def rational(q):return dict(numerator=str(q.numerator),denominator=str(q.denominator))
def save(n,v):
 p=O/n;p.write_text(json.dumps(plain(v),indent=2,allow_nan=False)+'\n');p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return dict(path=str(p),sha256=sha(p))
def arrays(n,**v):
 p=O/n;np.savez_compressed(p,**v);p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return dict(path=str(p),sha256=sha(p))

# Independent operation-by-operation absolute-error recurrence, all six dot
# associations. It does not import/reexecute the producer's gamma5 routine.
u=Q(1,2**53);eta=Q(1,2**1074)
def rounded(m,e):return m,e+u*(m+e)+eta
def dot_interval(m,e,c):
 terms=[rounded(abs(c[i])*m[i],abs(c[i])*e[i])for i in range(3)];results=[]
 for order in itertools.permutations(range(3)):
  a,b,c0=[terms[i]for i in order];s=rounded(a[0]+b[0],a[1]+b[1]);s=rounded(s[0]+c0[0],s[1]+c0[1]);results.append(s[1])
 return sum(x[0]for x in terms),max(results)
def independent_pipeline(P,T,R,RC,V,H):
 cm=[];ce=[]
 for row in RC:
  m,e=dot_interval([H]*3,[Q(0)]*3,row);cm.append(m);ce.append(e)
 tm=[];te=[];am=[];ae=[]
 for i in range(3):
  m,e=rounded(abs(T[i])+cm[i],ce[i]);tm.append(m);te.append(e)
  m,e=rounded(P[i]+m,e);am.append(m);ae.append(e)
 lm=[];le=[];oe=[]
 for k in range(3):
  m,e=dot_interval(am,ae,[R[j][k]for j in range(3)]);lm.append(m);le.append(e);oe.append(rounded(m+V[k],e)[1])
 return dict(local_error=le,rounded_offset_error=oe)

with(O/'normalized-BEFORE-AFTER-journal.jsonl').open('w')as log:
 def emit(x):journal(log,x)
 declaration=dict(schema='v2d.NEW-hard-training-source-conic-construction.v1',
  columns=454,world_translation_columns=453,clearance_phase_slack_column=453,
  world_translation_box_m=[-.02,.02],phase_slack_box_m=[0,.05],
  quadratic_regularization=1e-3,objective='rho + 0.5*1e-3*sum(world_delta^2)',
  source='ALL286608SAVED sufficient median proxies with independently justified outward arithmetic proposal guards',
  geometry='SAVED initialF union-maximizing axes ONLY, not rejected nonzero points; selected branch restriction',
  motion='ALL299 original max-linear-speed/acceleration affine SOC maps; actualp95 and angular invariants checked on raw proposal',
  hard_training_raster_median_quality_151_POSTconstruction_ONLY=True,
  original_fullpixel_heldout_source_gates_POSTconstruction_only_IF_training_and_motion_pass=True,
  original_tolerances_UNCHANGED=dict(source=1e-10,object_motion=1e-9,robot_position=1e-8,robot_motion=1e-7,image_px=1e-6,root_orientation=1e-9),
  no_solver_or_actual_nonzero_episode_pose_selected_by_this_builder=True,
  no_native_position_physics_GPU_PPO=True,all_original_F_robot_commands_and_object_rotations_FIXED=True,
  arithmetic_guards_NOT_actual_source_or_physical_acceptance=True)
 save('ONE-new-model-declaration-BEFORE-construction.json',declaration);emit(dict(kind='BEFORE_NEW_error_bindings_and_model',declaration=declaration))
 bounds=read(B/'all151-NEW-uniform-error-bindings.json');proofs=[]
 for b in bounds:
  frame=b['frame'];P=list(map(unpack,b['P_component_absolute']));T=list(map(unpack,b['T_translation']));V=list(map(unpack,b['V_component_absolute']));R=[list(map(unpack,r))for r in b['R_binary']];RC=[list(map(unpack,r))for r in b['RC_binary']];H=unpack(b['world_halfwidth'])
  future=independent_pipeline(P,T,R,RC,V,H);base=independent_pipeline(P,T,R,RC,V,Q(0));actual=b['arithmetic_bounds'];margins=[]
  for kind,own in [('future',future),('baseline',base)]:
   for key in ['local_error','rounded_offset_error']:
    for i in range(3):margins.append(unpack(actual[kind][key][i])-own[key][i])
  bias=[H*sum(abs(unpack(v))for v in row)for row in b['inherited_exact_map_errors']]
  independent=sum(future['local_error'][i]+base['rounded_offset_error'][i]+bias[i]for i in range(3));allow=unpack(actual['L1_distance_allowance'])
  margins.append(allow-independent);lo=Q(b['original_lower_radius']);hi=Q(b['original_upper_radius']);gl=Q(b['NEW_outward_proposal_lower_radius']);gh=Q(b['NEW_outward_proposal_upper_radius']);margins.extend([gl-lo-allow,gh-hi-allow])
  proof=dict(frame=frame,method='all6dotorder exact operation-wise recurrence; producer gamma5 NOT imported',
   minimum_exact_margin=rational(min(margins)),uniform_distance_bound=actual['L1_distance_allowance'],
   inherited124defects_reused_BY_HASH_NOT_reaudited=True,old_bound_file_sha256=sha(B/f'frame-{frame:03d}-NEW-uniform-proxy-error-RETURN.json'))
  save(f'frame-{frame:03d}-NEW-independent-roundoff-proof.json',proof);emit(dict(kind='AFTER_NEW_independent_roundoff_RETURN',actual_RETURN=proof));assert min(margins)>=0
  proofs.append(proof)
 save('all151-NEW-independent-roundoff-proofs.json',proofs)
 # Complete original-motion and cached-initial-F geometry bindings.
 caps=read(D/'runtime-config-and-original-authority.json')['original_object_caps']
 with np.load(D/'dynamic-axis-group-bindings.npz')as z:frames=z['pair_frames']
 initial=S/'e037-object-dynamic-evaluation-v1/evaluation-000-NEW-union-support-RETURN.npz'
 with np.load(initial)as z:active=z['active_axis_indices'];gaps=z['gap_proxy_m'];initial_delta=z['world_translation']
 axes=np.load(D/'ALL-retained-axis-world-coefficients.npy',mmap_mode='r');chosen=np.asarray(axes[active]).copy()
 assert np.array_equal(initial_delta,np.zeros((151,3))),'Only initialF cache allowed'
 motion=sparse.load_npz(D/'dynamic-motion-linear-map.npz').tocoo();motion0=np.load(D/'dynamic-motion-constants.npy').reshape(299,3)
 # Nonnegative rows: finite454variable box then17520 selected geometry planes.
 n=454;nn=2*n+len(frames);rr=[np.arange(n),n+np.arange(n)];cc=[np.arange(n),np.arange(n)];vv=[np.ones(n),-np.ones(n)]
 rhs=[np.r_[np.full(453,.02),.05],np.r_[np.full(453,.02),0.]]
 gr=np.repeat(np.arange(len(frames)),3);gc=(3*frames[:,None]+np.arange(3)).ravel()
 rr += [2*n+gr,2*n+np.arange(len(frames))];cc += [gc,np.full(len(frames),453)];vv += [-chosen.ravel(),-np.ones(len(frames))];rhs += [gaps+.001]
 row=nn;motion_start=row
 mr=4*(motion.row//3)+1+motion.row%3;rr.append(row+mr);cc.append(motion.col);vv.append(-motion.data)
 mb=np.column_stack([np.r_[np.full(150,caps['maximum_linear_speed_m_s']),np.full(149,caps['maximum_linear_acceleration_m_s2'])],motion0]);rhs.append(mb.ravel());row+=1196
 source_start=row;blocks=[]
 for f in range(151):
  emit(dict(kind='BEFORE_NEW_canonical_source_block',frame=f,row_start=row))
  old=M/f'frame-{f:03d}-NEW-proxy-SOC-maps-RETURN.npz';bd=bounds[f]
  with np.load(old)as z:A0=z['canonical_A'];b0=z['canonical_b'].copy();r=z['nominal_radius_m']
  newr=np.where(r==bd['original_lower_radius'],bd['NEW_outward_proposal_lower_radius'],bd['NEW_outward_proposal_upper_radius']);b0[:,0]=newr
  a=A0.reshape(-1,3);ir,ic=np.nonzero(a);rr.append(row+ir);cc.append(3*f+ic);vv.append(a[ir,ic]);rhs.append(b0.ravel())
  br=arrays(f'frame-{f:03d}-NEW-guarded-source-rhs-RETURN.npz',b=b0,original_radius=r,NEW_outward_proposal_radius=newr)
  block=dict(frame=f,row_start=row,rows=len(b0)*4,SOC4=len(b0),column_start=3*f,old_map=str(old),old_map_sha256=sha(old),NEW_rhs=br,guard_file_sha256=sha(B/f'frame-{f:03d}-NEW-uniform-proxy-error-RETURN.json'))
  emit(dict(kind='AFTER_NEW_canonical_source_block_RETURN',actual_RETURN=block));blocks.append(block);row+=len(b0)*4
 A=sparse.coo_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),shape=(row,n)).tocsc();b=np.concatenate(rhs);q=np.r_[np.zeros(453),1.];P=sparse.diags(np.r_[np.full(453,1e-3),0.],format='csc')
 sparse.save_npz(O/'canonical-A.npz',A);sparse.save_npz(O/'canonical-P.npz',P);vectors=arrays('canonical-vectors.npz',b=b,q=q)
 meta=dict(schema='v2d.NEW-hard-training-source-canonical-model.v1',columns=n,rows=row,nonnegative_rows=nn,motion_SOC4=299,source_SOC4=sum(x['SOC4']for x in blocks),SOC4=299+sum(x['SOC4']for x in blocks),motion_start=motion_start,source_start=source_start,source_blocks=blocks,nonzeros=A.nnz,objective_regularization=1e-3,original_caps=caps,cached_initial_F_union_RETURN=str(initial),cached_initial_F_union_RETURN_sha256=sha(initial),old_maps_AND_all_radii_UNALTERED=True,new_arithmetic_guards_proposal_only=True,no_solver_called=True)
 save('canonical-metadata.json',meta);emit(dict(kind='AFTER_NEW_COMPLETE_canonical_constructor_RETURN',actual_RETURN=meta,canonical_hashes={n:sha(O/n)for n in ['canonical-A.npz','canonical-P.npz','canonical-vectors.npz']}))
 assert A.shape[0]==len(b)and row==1166056 and meta['SOC4']==286907
 report=dict(schema='v2d.NEW-hard-training-source-model-report.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),independent_uniform_roundoff_bindings=151,old124defects_AND_radii_retained=True,rows=row,columns=n,SOC4=meta['SOC4'],nonnegative_rows=nn,source_SOC4=meta['source_SOC4'],source_support_SCOPE='outward sufficient-proxy proposal restriction; actual source gates separate',no_solver_or_actual_nonzero_episode_pose=True,no_original_source_NN_control_or_model_producer_replay=True,original_gates_parameters_source_missing_F_UNCHANGED=True)
 save('report.json',report);emit(dict(kind='AFTER_NEW_model_report_RETURN',actual_RETURN=report));print(json.dumps(report))
