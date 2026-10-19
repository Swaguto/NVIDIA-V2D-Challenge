"""NEW source-certified continuous angle regions and one bounded MILP.

No new wrist mode, source observations, sampled graph recurrence, homotopy,
gradient direction or old optimizer return is used. All source points reused
from failed whole families were independently verified in their POINT scope.
"""
from pathlib import Path
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-index-continuous-source-box-path-v1'
sys.path.insert(0,str(S/'compute'))
import source_orientation_context_v1 as c
import thumb_conic_geometry_v1 as g
from planar_support_critical_points_v1 import point
from continuous_source_box_path_v1 import continuous_box_path
from scipy.spatial.transform import Rotation
np=c.np;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic()
j=(O/'all-continuous-source-box-proposals-and-MILP-return.jsonl').open('w')
controls=[]
for name,bank,expect in [
    ('continuous_interval_connects_when_centers_fail',[[[[0,0]]],[[[.04,.08]]],[[[.1,.1]]]],True),
    ('forced_adjacent_conflict_is_not_optimizer_success',[[[[0,0]]],[[[.12,.13]]],[[[.25,.25]]]],False),
]:
    result=continuous_box_path(bank,np.zeros((3,1)),[0,.1,.2],[-1.],[1.],[10.],time_limit=5.,node_limit=100)
    g.journal(j,dict(kind='new_analytic_continuous_box_path_return',name=name,result=result))
    passed=bool(result['success']==expect)
    if expect:
        p=np.array(result['proposed_angles']);passed &= bool(abs(np.diff(p,axis=0)).max()<=.05+1e-8 and abs(np.diff(p,n=2,axis=0)).max()<=.05+1e-8)
    controls.append(dict(name=name,passed=passed));assert passed,name

def inner_box(center,H,pixel,cap,lower,upper):
    center=np.array(center);C=H@np.r_[point(*center),1.]
    B=np.array([[c.K[0,0],0,c.K[0,2]-pixel[0]],[0,c.K[1,1],c.K[1,2]-pixel[1]]])
    radius=cap+1e-6
    f=float(np.linalg.norm(B@C)-radius*C[2])
    metric_guard=1e-10
    guard=(np.linalg.norm(B,2)+radius)*metric_guard
    movement=(np.linalg.norm(B@H[:,:2],2)+radius*np.linalg.norm(H[2,:2]))
    source_rho=max(0.,(-f-guard)/(2*max(movement,1e-30)))
    depth_rho=max(0.,(C[2]-.01-metric_guard)/(2*max(np.linalg.norm(H[2,:2]),1e-30)))
    rr=min(.05,source_rho/(.0458+2*.035),depth_rho/(.0458+2*.035))
    box=np.c_[np.maximum(lower,center-rr),np.minimum(upper,center+rr)]
    status='continuous_source_certified_inner_box'
    if f>0 or C[2]<=.01 or (center<lower).any() or (center>upper).any():
        status='rejected_center_outside_exact_constructor_source_or_position'
    elif rr<=0: status='verified_source_singleton_no_positive_inner_radius'
    delta=np.maximum(abs(box[:,0]-center),abs(box[:,1]-center))
    rho=.0458*delta[0]+.035*sum(delta)
    certificate=dict(camera_center=C,source_cone_center_residual=f,source_radius_including_ORIGINAL_tolerance_px=radius,
        camera_cone_Lipschitz=movement,metric_guard_m=metric_guard,source_cone_guard=guard,
        maximum_planar_movement_m=rho,source_cone_upper_bound=f+movement*rho+guard,
        positive_depth_lower_bound=C[2]-np.linalg.norm(H[2,:2])*rho-metric_guard,
        construction_radius_rad=rr,classification=status,
        numerical_analytic_bound_not_formal_directed_rounding_proof=True)
    return box,certificate

# New analytic enclosure checks use independent exact camera projections over
# finite angular samples. They do not establish formal directed rounding.
rng=np.random.default_rng(190109)
for center in [[.1,.7],[.7,1.0]]:
    H=np.array([[1.,0.,.1],[0.,1.,-.1],[.1,-.2,.5]])
    C=H@np.r_[point(*center),1.];pix=c.uv(C[None])[0];box,cert=inner_box(center,H,pix,20.,[-1.,0.],[1.,2.])
    values=[]
    for a,b in rng.uniform(box[:,0],box[:,1],size=(128,2)):
        cam=H@np.r_[point(a,b),1.];values.append(float(np.linalg.norm(c.uv(cam[None])[0]-pix)))
    passed=bool(cert['source_cone_upper_bound']<0 and max(values)<=20.+1e-6)
    g.journal(j,dict(kind='new_analytic_source_inner_box_control',center=center,box=box,certificate=cert,independent_source_errors_px=values,passed=passed))
    controls.append(dict(name='independent_camera_cone_enclosure_'+str(center),passed=passed));assert passed

bank_path=S/'e037-integrated-source-corpora-reachability-v1/all-factorized-saved-banks-and-recurrence-returns.jsonl'
assert sha(bank_path)==json.loads((bank_path.parent/'report.json').read_text())['all_outputs_sha256'][bank_path.name]
points={}
for line in bank_path.read_text().splitlines():
    row=json.loads(line)
    if row.get('kind')=='reused_all_source_points_without_Cartesian_pruning' and row.get('digit')=='index':points[row['frame']]=list(row['bank'])
new_path=S/'e037-index-temporal-box-independent-v1/verified-new-index-triples.json'
assert sha(new_path)==json.loads((new_path.parent/'report.json').read_text())['all_outputs_sha256'][new_path.name]
for row in json.loads(new_path.read_text())['verified']:points[109].append(row['angles_rad'])
with np.load(S/'e037-strict-source-wrist-mode-v1/UNQUALIFIED-strict-source-root-proposal.npz') as z:
    commands=z['robot_qpos'].copy();weights=z['weights'].copy()
source_rows={v['frame']:v for v in json.loads((S/'e037-thumb-ray-independent-validation-v1/report.json').read_text())['source_rows'] if v['hand']=='right'}
banks=[];records=[];counts={};reused=0
for frame in range(len(c.q)):
    rowbanks=[]; rowrecords=[]
    if weights[frame]<=1e-14:
        box=np.c_[commands[frame,1,7:9],commands[frame,1,7:9]]
        rowbanks=[box];rowrecords=[dict(classification='unchanged_saved_source_singleton',box=box)]
    else:
        camera=c.cameras[frame];pixel=c.uv((c.human[frame,1,[8]]-camera[:3,3])@camera[:3,:3])[0]
        A=camera[:3,:3].T@c.stage.T;R=Rotation.from_quat(commands[frame,1,3:7]).as_matrix()
        H=np.c_[A@R@np.eye(3)[:,:2],A@(commands[frame,1,:3]+R@np.array([.0777,-.0016,.0285]))-camera[:3,:3].T@camera[:3,3]]
        cap=source_rows[frame]['source_before_after_px'][0][2];lower,upper=c.bounds(frame)
        unique={tuple(np.round(v,12)):np.array(v) for v in points[frame]}
        for center in unique.values():
            g.journal(j,dict(kind='BEFORE_new_continuous_source_inner_box',frame=frame,center=center,H=H,pixel=pixel,original_radius_px=cap))
            box,certificate=inner_box(center,H,pixel,cap,lower[:2],upper[:2])
            record=dict(frame=frame,center=center,box=box,H=H,pixel=pixel,ORIGINAL_radius_px=cap,certificate=certificate)
            g.journal(j,dict(kind='AFTER_new_continuous_source_inner_box',record=record));rowrecords.append(record)
            counts[certificate['classification']]=counts.get(certificate['classification'],0)+1
            if certificate['classification'].startswith('rejected'):continue
            rowbanks.append(box)
    banks.append(np.array(rowbanks).reshape(-1,2,2));records.append(dict(frame=frame,box_count=len(rowbanks),proposals=rowrecords))
g.journal(j,dict(kind='BEFORE_one_NEW_continuous_source_box_MILP',banks=banks,
    original_reference=c.aligned[:,1,7:9],timestamps=c.ts,original_actual_position_limits=c.limits[:2],
    velocity_limits=[12.,12.],component=.25,adjacent_added=.05,second_added=.05,
    time_limit_s=60,node_limit=5000,objective='minimum_total_L1_correction_against_original_aligned_reference',
    prior_graph_recursion_NOT_executed=True))
result=continuous_box_path(banks,c.aligned[:,1,7:9],c.ts,c.limits[:2,0],c.limits[:2,1],np.ones(2)*12)
g.journal(j,dict(kind='AFTER_one_NEW_continuous_source_box_MILP_RETURN',result=result))
(O/'continuous-box-MILP-return.json').write_text(json.dumps(g.plain(result),allow_nan=False,indent=2)+'\n')
candidate=None;motion=None
if result['proposed_angles'] is not None:
    angles=np.array(result['proposed_angles']);d=angles-c.aligned[:,1,7:9]
    motion=dict(component_rad=float(abs(d).max()),added_adjacent_rad=float(abs(np.diff(d,axis=0)).max()),
        added_second_rad=float(abs(np.diff(d,n=2,axis=0)).max()),maximum_URDF_velocity_rad_s=float((abs(np.diff(angles,axis=0))/np.diff(c.ts)[:,None]).max()))
    g.journal(j,dict(kind='returned_candidate_direct_original_motion',motion=motion))
    with np.load(S/'e037-saved-middle-source-component-v1/UNQUALIFIED-middle-source-motion-component.npz') as z:candidate=z['robot_qpos'].copy()
    candidate[:,1,7:9]=angles
    np.savez_compressed(O/'UNQUALIFIED-index-middle-source-motion-proposal.npz',robot_qpos=candidate)
j.close()
(O/'all-continuous-source-box-records.json').write_text(json.dumps(g.plain(records),allow_nan=False,indent=2)+'\n')
out=dict(schema='v2d.continuous-index-source-box-path.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    producer_sha256=sha(__file__),new_analytic_controls=controls,source_inner_box_classifications=counts,
    total_inner_boxes=sum(len(v) for v in banks),empty_rows=[i for i,v in enumerate(banks) if not len(v)],
    MILP_return_success=result['success'],MILP_status=result['status'],MILP_message=result['message'],
    returned_proposal_exists=candidate is not None,returned_direct_original_motion=motion,
    finite_continuous_inner_family_NOT_global_unreachability=True,source_authority_unchanged=True,
    no_old_graph_or_source_constructor_repeated=True,full_source_URDF_native_independent_checks_pending=candidate is not None,
    current_verified_reference_unchanged=True,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,
    whole_reference_or_task_qualified=False,all_outputs_sha256={p.name:sha(p) for p in O.iterdir() if p.is_file() and p.name not in ['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(g.plain(out)))
