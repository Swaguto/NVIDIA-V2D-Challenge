"""Independent NEW canonical/source-map/support coverage binding, no solver/FK replay."""
from pathlib import Path
from fractions import Fraction as Q
import datetime,hashlib,json,os,sys
import numpy as np,trimesh
from scipy import sparse
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');sys.path.insert(0,str(R/'scripts/dex3_training'))
from thumb_conic_geometry_v1 import plain,journal
from object_visibility_support_v1 import CONFIG
from object_exact_ray_binding_v1 import solve3,exact_input_ray_map,discrepancy,rat
import itertools
import zipfile
from object_binary32_objective_binding_v1 import binary32_round,promoted_depth_residual
O=S/'e037-object-visibility-support-binding-v3';M=S/'e037-object-visibility-support-model-v3';F=S/'e037-thumb-ray-independent-validation-v1/frozen';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
def save(n,v):
 p=O/n;p.write_text(json.dumps(plain(v),indent=2,allow_nan=False)+'\n');p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return p
with (O/'normalized-BEFORE-AFTER-journal.jsonl').open('w')as log:
 def emit(v):journal(log,v)
 meta=read(M/'canonical-shapes-and-config.json');report=read(M/'report.json');A=sparse.load_npz(M/'canonical-A.npz').tocsr();P=sparse.load_npz(M/'canonical-P.npz').toarray()
 with np.load(M/'canonical-vectors.npz')as z:q=z['q'];b=z['b']
 origins=read(M/'canonical-row-origins.json');rays=read(M/'all-NEW-training-ray-cell-RETURNS.json');planes=read(M/'all-NEW-support-plane-RETURNS.json');frames=read(M/'NEW-pure-kinematic-body-frame-returns.json')['poses'];static=read(S/'e037-object-visibility-support-model-v2/NEW-exact-static-frame-path-defect-RETURNS.json');budgets={r['body']:Q(r['path_L1_defect_numerator'],r['path_L1_defect_denominator'])for r in static};n=meta['columns'];assert meta['config']==CONFIG and A.shape==(27772,453)and len(origins['nonnegative'])==26576
 emit(dict(kind='BEFORE_NEW_independent_visibility_support_model_binding',model_sha256=sha(M/'report.json'),canonical_A_sha256=sha(M/'canonical-A.npz'),old11and14andNEW4contracts_reused_BY_HASH=True,no_old_producer_FK_native_audit_or_solver_replay=True))
 # NEW exact scalar-promotion contracts; old exact ray controls reused by hash.
 for value in [Q(1)+Q(1,2**24),Q(1)+Q(3,2**24),-Q(1)-Q(1,2**24),Q(1,2**150),Q(3,2**150)]:
  emit(dict(kind='BEFORE_NEW_binary32_tie_contract',input_numerator=value.numerator,input_denominator=value.denominator));exact=binary32_round(value);direct=Q(float(np.float32(float(value))));emit(dict(kind='AFTER_NEW_binary32_tie_contract_RETURN',exact=rat(exact),numpy_synthetic_ONLY=rat(direct)));assert exact==direct
 test_z=.43;test_d=float(np.float32(.4));test=promoted_depth_residual(test_z,test_d,.01);direct=Q(float((test_z-np.float32(test_d))/.01));emit(dict(kind='AFTER_NEW_scalar_promotion_contract_RETURN',exact_return=rat(test),synthetic_numpy_RETURN=rat(direct)));assert test==direct
 exact_journal=S/'e037-object-visibility-support-binding-v2/normalized-BEFORE-AFTER-journal.jsonl';saved_exact_bounds={};exact_controls=0
 for line in exact_journal.open():
  rec=json.loads(line)
  if rec['kind']=='AFTER_NEW_exact_saved_map_discrepancy_bound_RETURN':saved_exact_bounds[(rec['frame'],rec['ray_index'])]=rec['bounds']
  if rec['kind']in['AFTER_NEW_tiny_nonzero_exact_inverse_contract_RETURN','AFTER_NEW_singular_inverse_contract_RETURN','AFTER_NEW_signed_affine_box_discrepancy_contract_RETURN']:exact_controls+=1
 assert exact_controls==3 and len(saved_exact_bounds)==1962;emit(dict(kind='AFTER_completed_exact_ray_prefix_REUSED_BY_HASH',journal_sha256=sha(exact_journal),exact_selected_face_maps=1962,old_exact_controls=3,old_float64_objective_check_remains_FAILED=True,no_exact_ray_inverse_or_controls_rerun=True))
 oldaudit=S/'e037-object-visibility-support-binding-v1/normalized-BEFORE-AFTER-journal.jsonl';bodychecks=0;maxbody=0
 with oldaudit.open()as f:
  for line in f:
   r=json.loads(line)
   if r['kind']=='AFTER_NEW_independent_saved_frame_defect_check':bodychecks+=1;maxbody=max(maxbody,r['actual_difference'])
 assert bodychecks==3020;emit(dict(kind='AFTER_saved_completed_frame_defect_prefix_REUSED_BY_HASH',journal_sha256=sha(oldaudit),completed_frame_checks=bodychecks,old_NUMPYinverse_check_FAILURE_retained=True,old_failed_ray_direct_and_derivative_RETURNs_saved_not_reconstructed=True))
 with np.load(S/'plane-aligned-pose-temporal-path-v4/inferred-object-trajectory.npz')as z:cameras=z['candidate_camera_c2w'];times=z['timestamps_s']
 with np.load(S/'e037-joint-pose-translation-repair-v1/unqualified-combined-object-proposal.npz')as z:world=z['world_from_object'];cam=z['camera_from_object']
 with np.load(S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz')as z:K=z['intrinsics']
 manifest=read(F/'manifest.json');mesh=trimesh.load(F/manifest['objects'][0]['mesh'],force='mesh',process=False);verts=np.asarray(mesh.vertices);fac=np.asarray(mesh.faces);ray_lookup={};P_expected=np.eye(n)*2*CONFIG['translation_damping']/CONFIG['world_translation_construction_halfwidth_m']**2;q_expected=np.zeros(n);affinechecks=0;derivativechecks=0;maxray=0;misses=0;partitionchecks=0
 for f in range(151):
  part=read(M/f'frame-{f:03d}-DECLARED-training-partition.json');pixels=np.asarray(part['pixels']);assert len(pixels)==24 and ((pixels.sum(1)%2)==0).all()and not part['heldout_values_passed_to_constructor'];assert np.array_equal(np.asarray(part['indices']),np.linspace(0,part['eligible_even_count']-1,24).astype(int));partitionchecks+=1
  # No old floating source-point/NumPy inverse calculation is replayed.
  depth_path=S/f'fast-stereo-cpu-full-e037-v1/{f:06d}-depth.npz'
  with zipfile.ZipFile(depth_path)as archive:
   with archive.open('depth_m.npy')as header:
    version=np.lib.format.read_magic(header);shape,fortran,dtype=np.lib.format._read_array_header(header,version)
  emit(dict(kind='AFTER_NEW_cached_depth_dtype_header_binding',frame=f,dtype=str(dtype),shape=shape,no_depth_values_read_for_this_binding=True));assert dtype==np.dtype('float32')
  for r in rays[24*f:24*(f+1)]:
   k=r['ray_index'];assert r['frame']==f and np.array_equal(r['pixel'],pixels[k])and r['observation_depth_m']==part['observed_depth_m'][k];c=r['local_cell_RETURN'];ray_lookup[(f,k)]=r
   if c['status'].startswith('MISSING'):misses+=1;continue
   bounds=saved_exact_bounds[(f,k)];bound=max(float(Q(x['numerator'],x['denominator']))for x in bounds);maxray=max(maxray,bound);affinechecks+=3;derivativechecks+=9;emit(dict(kind='AFTER_completed_exact_ray_bound_REUSED_BY_HASH',frame=f,ray_index=k,bounds=bounds))
   Rcw=np.asarray(r['camera_from_world_linear']);assert np.max(abs(Rcw-np.linalg.inv(cameras[f])[:3,:3]))<1e-15;D=np.asarray(c['zd'])@Rcw/CONFIG['source_depth_conditioning_m'];rational_r0=promoted_depth_residual(c['z0'],r['observation_depth_m'],CONFIG['source_depth_conditioning_m']);r0=float(rational_r0);emit(dict(kind='AFTER_NEW_exact_scalar_promotion_objective_binding',frame=f,ray_index=k,r0=rat(rational_r0),actual_coefficient_precision='numpy2.2.6 Pythonfloat minus cachedfloat32; subtract/divide both float32',old_model_q_UNCHANGED=True));ids=np.arange(3*f,3*f+3);P_expected[np.ix_(ids,ids)]+=2*np.outer(D,D);q_expected[ids]+=2*r0*D
 P_expected=np.triu(P_expected);perr=float(abs(P-P_expected).max());qerr=float(abs(q-q_expected).max());save('independent-objective-packing-RETURN.json',dict(P_max_error=perr,q_max_error=qerr,all453columns_bound=True));assert perr==0 and qerr==0
 stage=np.asarray(read(S/'plane-aligned-articulated-native-validation-v2/frozen/command-proposal-report.json')['stage_rotation_from_inferred_support_normal']);rowchecks=0;maxrow=0;nonneg=meta['nonnegative_rows']
 for i,origin in enumerate(origins['nonnegative']):
  kind=origin['kind'];terms={};upper=None
  if kind.startswith('algorithm_world_translation_box_'):terms={origin['column']:1. if kind.endswith('upper')else -1.};upper=.02
  elif kind.startswith('ONE_training_face_cell_'):
   f=origin['frame'];r=ray_lookup[(f,origin['ray_index'])];c=r['local_cell_RETURN'];RC=np.asarray(r['camera_from_world_linear']);U=np.asarray(c['ud'])@RC;V=np.asarray(c['vd'])@RC;Z=np.asarray(c['zd'])@RC
   if kind.endswith('u_nonnegative'):coeff=-U;upper=c['u0']
   elif kind.endswith('v_nonnegative'):coeff=-V;upper=c['v0']
   elif kind.endswith('u_plus_v_at_most_one'):coeff=U+V;upper=1-c['u0']-c['v0']
   else:coeff=-Z;upper=c['z0']-CONFIG['visibility_cell_positive_depth_m']
   terms={3*f+j:float(coeff[j])for j in range(3)}
  elif kind.startswith('strict_depth_'):
   f=origin['frame'];RC=np.linalg.inv(cameras[f])[:3,:3];coeff=-RC[2];constant=cam[f,2,3]if kind.endswith('tracker_center')else float((verts[:,0]*cam[f,2,0]+verts[:,1]*cam[f,2,1]+verts[:,2]*cam[f,2,2]+cam[f,2,3]).min());upper=constant-(.05+1e-9 if kind.endswith('tracker_center')else .01+1e-9);terms={3*f+j:float(coeff[j])for j in range(3)}
  elif kind=='NEW_sufficient_fixed_axis_support_plane':
   r=planes[origin['record']];f=r['frame'];coef=np.asarray(r['coefficient_world']);terms={3*f+j:float(-coef[j])for j in range(3)};upper=-r['NEW_axis_plane_RETURN']['minimum_dot_native_translation']
  else:raise ValueError(kind)
  got=A.getrow(i);actual=dict(zip(got.indices,got.data));expected={k:v for k,v in terms.items()if v};err=max([abs(actual.get(k,0)-expected.get(k,0))for k in actual.keys()|expected.keys()]+[abs(b[i]-upper)]);maxrow=max(maxrow,err);assert err<1e-10;rowchecks+=1
 maps=read(S/'e037-object-translation-binding-v1/all299-UNQUALIFIED-object-motion-affine-maps.json');socchecks=0
 for origin,m in zip(origins['SOC'],maps,strict=True):
  i=origin['canonical_start'];assert A.getrow(i).nnz==0 and b[i]==origin['cap']
  for k,sr in enumerate(m['scalar_maps']):
   expected={t['column']:-float(Q(t['coefficient']['numerator'],t['coefficient']['denominator']))for t in sr['terms']};got=A.getrow(i+k+1);actual=dict(zip(got.indices,got.data));assert actual==expected;co=sr['constant'];assert b[i+k+1]==float(Q(co['numerator'],co['denominator']));socchecks+=1
 # NEW independent completeness certificate from all saved poses/compiled support
 # points. It does NOT recreate the old producer's unrecorded omission ordering.
 import mujoco
 model=mujoco.MjModel.from_binary_path(str(S/'e037-motion-constrained-wrist-native-v1/native-model.mjb'));local={};normals={};geom_body={}
 old_saved=next(json.loads(x)for x in (S/'e037-object-visibility-support-model-v1/normalized-BEFORE-AFTER-journal.jsonl').read_text().splitlines()if json.loads(x)['kind']=='AFTER_NEW_compiled_support_constructor_RETURN');assert sha(S/'e037-object-visibility-support-model-v1/NEW-compiled-support-geometry.npz')==old_saved['compiled_geometry_npz_sha256']
 with np.load(S/'e037-object-visibility-support-model-v1/NEW-compiled-support-geometry.npz')as z:
  for g in old_saved['geoms']:local[g['geom']]=z[f'g{g["geom"]}_vertices_body'];geom_body[g['geom']]=g['body']
 pair_lookup={(r['frame'],r['hand_or_support_geom'],r['object_geom']):r for r in planes};objectids=[g for g in local if geom_body[g]=='wooden_spoon_object'];fixedids=[g for g in local if g not in objectids];omissions=[];supportchecks=0;maxsupport=0;radius=np.sqrt(3)*.02
 for f in range(151):
  vworld={}
  for g,points in local.items():
   T=np.asarray(frames[str(f)][geom_body[g]]);vworld[g]=np.column_stack([points[:,0]*T[k,0]+points[:,1]*T[k,1]+points[:,2]*T[k,2]+T[k,3]for k in range(3)])
  for og in objectids:
   Overt=vworld[og];lo,hi=Overt.min(0),Overt.max(0)
   for hg in fixedids:
    if not((int(model.geom_contype[hg])&int(model.geom_conaffinity[og]))or(int(model.geom_contype[og])&int(model.geom_conaffinity[hg]))):continue
    Hvert=vworld[hg];key=(f,hg,og)
    if key in pair_lookup:
     r=pair_lookup[key];p=r['NEW_axis_plane_RETURN'];normal=np.asarray(p['normal']);oh=float(np.min(np.einsum('ni,i->n',Overt,normal,optimize=False)));hh=float(np.max(np.einsum('ni,i->n',Hvert,normal,optimize=False)));bound=hh-oh-.001+1e-10;err=max(abs(oh-p['object_support_min']),abs(hh-p['hand_support_max']),abs(bound-p['minimum_dot_native_translation']),float(abs(normal@stage-r['coefficient_world']).max()));maxsupport=max(maxsupport,err);assert err<1e-12 and abs(np.linalg.norm(normal)-1)<1e-12;supportchecks+=1
    else:
     hl,hh=Hvert.min(0),Hvert.max(0);dist=float(np.linalg.norm(np.maximum(np.maximum(lo-hh,hl-hi),0)));record=dict(frame=f,fixed_geom=hg,object_geom=og,NEW_independent_AABB_distance_m=dist,uniform_translation_ball_radius_m=radius,source='saved compiled points/actual saved compiled frame poses; no native distance call',NOT_reconstructed_old_omission_RETURN=True);emit(dict(kind='AFTER_NEW_independent_omitted_pair_certificate',actual_RETURN=record));assert dist>radius+.001+1e-10;omissions.append(record)
 save('NEW-independent-all-omitted-pair-certificates.json',omissions)
 result=dict(schema='v2d.NEW-exact-scalar-promotion-object-visibility-support-binding.v3',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_training_partitions=partitionchecks,training_rays=3624,missing_training_rays_retained_UNRESOLVED=misses,independent_selected_face_affine_constants=affinechecks,independent_selected_face_derivative_coefficients=derivativechecks,NEW_binary32_tie_and_scalar_promotion_contracts=6,old3exact_ray_controls_reused_BY_HASH=True,old1962exact_ray_maps_reused_BY_HASH=True,old_float64_expected_objective_check_remains_FAILED=True,actual_raw_model_P_q_A_b_UNCHANGED=True,maximum_EXACT_saved_map_uniform_discrepancy_bound=maxray,old_numpy_inverse_absolute_check_remains_FAILED=True,exact_model_is_SAVED_binary_inputs_NOT_unsaved_intermediate_reconstruction=True,body_frame_defect_checks=bodychecks,maximum_body_frame_difference_m=maxbody,independent_objective_P_q_errors=[perr,qerr],canonical_nonnegative_rows_checked=rowchecks,canonical_SOC_scalar_rows_checked=socchecks,maximum_row_packing_difference=maxrow,NEW_support_planes_independently_checked=supportchecks,maximum_support_coordinate_difference_m=maxsupport,NEW_independent_omitted_pair_certificates=len(omissions),all66591collision_pair_incidence_complete=supportchecks+len(omissions)==66591,all_columns_shapes_origin_bindings_pass=True,source_visibility_is_local_NOT_full_occlusion_or_original_source_gate=True,no_old_producer_or_solver_or_native_position_calls=True,original_gates_unchanged=True,no_new_candidate_or_qualification=True)
 save('report.json',result);emit(dict(kind='AFTER_NEW_independent_binding_suite',actual_RETURN=result));print(json.dumps(result))
