"""NEW load-span diagnosis on saved records only; no native/LP job reruns."""
from pathlib import Path
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');sys.path.insert(0,str(R/'scripts/dex3_training'))
import numpy as np
from v2d_dex3_flashchord.contact_load_feasibility import ContactGeometry
O=S/'e037-saved-contact-wrench-span-v1';start=time.monotonic();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();plan_path=S/'compute/saved-contact-wrench-span-CPU-plan-v1.json';plan=json.loads(plan_path.read_text())
def matrix(contacts,com):
 W=np.zeros((6,sum(c.dimension for c in contacts)));at=0
 for c in contacts:
  n=min(c.dimension,3);basis=c.frame_into_object_world;W[:3,at:at+n]=basis[:n].T;W[3:,at:at+n]=np.cross(c.point_world_m-com,basis[:n]).T
  if c.dimension>=4:W[3:,at+3]=basis[0]
  if c.dimension==6:W[3:,at+4:at+6]=basis[1:].T
  at+=c.dimension
 return W
scale=np.array([1.,1.,1.,10.,10.,10.])
def span(contacts,com,load):
 W=matrix(contacts,com);A=scale[:,None]*W;load=np.asarray(load,float);scaled=scale*load
 if A.shape[1]:U,s,vh=np.linalg.svd(A,full_matrices=True);rank=int(np.sum(s>1e-10*max(1,float(s[0]))))
 else:U=np.eye(6);s=np.empty(0);rank=0
 residual=scaled-U[:,:rank]@(U[:,:rank].T@scaled);relative=float(np.linalg.norm(residual)/max(1e-12,np.linalg.norm(scaled)))
 return dict(wrench_span_rank=rank,full_six_dimensional_span=rank==6,required_wrench_in_unrestricted_span=relative<=1e-8,scaled_relative_residual=relative,residual_wrench_world=(residual/scale).tolist(),singular_values_scaled=s.tolist(),force_sign_friction_and_motor_constraints_all_ignored=True,span_membership_does_not_establish_force_feasibility_or_grasp=True)
def point(p):return ContactGeometry(np.asarray(p,float),np.eye(3),3,np.ones(5))
controls=[]
for label,contacts,load,rank,within in [
 ('one_point_force',[point([0,0,0])],[0,0,1,0,0,0],3,True),
 ('one_point_pure_moment',[point([0,0,0])],[0,0,0,1,0,0],3,False),
 ('two_point_axial_moment',[point([-.03,0,0]),point([.03,0,0])],[0,0,0,1,0,0],5,False),
 ('three_noncollinear_points',[point([0,0,0]),point([.03,0,0]),point([0,.03,0])],[1,2,3,4,5,6],6,True)]:
 r=span(contacts,np.zeros(3),load);assert r['wrench_span_rank']==rank and r['required_wrench_in_unrestricted_span']==within;controls.append(dict(label=label,result=r))
path=S/'e037-common-root-atlas-native-v1/report.json';saved=json.loads(path.read_text());meta_path=S/'compute/palm-wrench-library-compatibility-inputs-v1.json';meta=json.loads(meta_path.read_text());load_path=Path(meta['source_report']);lr=json.loads(load_path.read_text());loads={k:lr['rows'][0][field] for k,field in [('gravity','gravity_load_wrench_world'),('estimated_motion','estimated_reference_motion_load_wrench_world')]};com=np.asarray(meta['object_COM_native_world_m']);rows=[]
for trial in saved['rows']:
 if not trial['geometric_LP']['geometrically_feasible']:continue
 contacts=[ContactGeometry(c['point_world_m'],c['frame_into_object_world'],c['dimension'],c['friction']) for c in trial['actual_right_hand_object_contacts']];results={k:span(contacts,com,w) for k,w in loads.items()};rows.append(dict(candidate_id=trial['candidate_id'],native_contact_count=len(contacts),native_contact_dimensions=[c.dimension for c in contacts],native_body_pairs=[c['native_bodies'] for c in trial['actual_right_hand_object_contacts']],load_span_results=results,saved_actual_effort_load_checks_failed=all(not c['feasible_in_outer_model'] for c in trial['actual_effort_load_checks'].values()),saved_penetration_m=trial['maximum_single_knot_penetration_m']))
report=dict(schema='v2d.saved-contact-required-wrench-span-audit.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),plan_sha256=sha(plan_path),inputs_sha256={str(p):sha(p) for p in [path,meta_path,load_path]},analytic_controls=controls,rows=rows,summary=dict(saved_native_contact_sets=len(rows),both_required_loads_outside_even_unrestricted_span=sum(all(not a['required_wrench_in_unrestricted_span'] for a in r['load_span_results'].values()) for r in rows),full_rank_but_saved_constrained_load_checks_still_fail=sum(all(a['full_six_dimensional_span'] for a in r['load_span_results'].values()) for r in rows)),nominal_model_mass_friction_and_inferred_motion_not_measured_truth=True,no_condim_or_other_model_parameter_changed=True,no_native_load_LP_or_optimizer_rerun=True,no_force_budget_invented=True,no_physics_or_GPU_work=True,new_PPO_steps=0,training_approved=False,physical_task_accepted=False,wall_seconds=time.monotonic()-start);(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
