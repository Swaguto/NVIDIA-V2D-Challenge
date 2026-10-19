"""Independent actual-URDF/source/triple checks of NEW saved 109 witnesses.

No chart solve or old graph recurrence repeats. The five unresolved boxes
receive new source-cone dual support diagnostics, never an empty inference
from the absence of sampled boundary points.
"""
from pathlib import Path
import datetime, hashlib, json, sys, time
S=Path(__file__).resolve().parents[1]; O=S/'e037-index-temporal-box-independent-v1'
sys.path.insert(0,str(S/'compute'))
import source_orientation_context_v1 as c
import thumb_conic_geometry_v1 as g
from planar_support_critical_points_v1 import support, point
from scipy.spatial.transform import Rotation
from v2d_dex3_flashchord.kinematics import URDFKinematics
np=c.np; sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest(); start=time.monotonic()
P=S/'e037-index-temporal-source-boxes-v1'; path=P/'all-correlated-index-source-domains.json'
assert sha(path)==json.loads((P/'report.json').read_text())['all_outputs_sha256'][path.name]
saved=json.loads(path.read_text()); H=np.array(saved['H']); pixel=np.array(saved['source_pixel']); cap=saved['original_source_radius_px']
with np.load(S/'e037-strict-source-wrist-mode-v1/UNQUALIFIED-strict-source-root-proposal.npz') as z: commands=z['robot_qpos'].copy()
kin=URDFKinematics(c.F/'robot/urdfs/g1/dex3_right.urdf')
j=(O/'all-independent-source-triples-and-support-witnesses.jsonl').open('w')
verified=[]; rejected=[]; maximum=0.; witness_checks=0; support_rows=[]
for row in saved['records']:
    if not row['source_chart']: continue
    pair=np.array(row['pair']); pair_ids=row['pair_ids']
    for w in row['source_chart']['witnesses']:
        angles=np.array(w['angles_rad']); cmd=commands[109,1].copy();cmd[7:9]=angles
        g.journal(j,dict(kind='BEFORE_independent_new_URDF_source_triple',pair_ids=pair_ids,pair=pair,command=cmd))
        tip=kin.poses(cmd[:7],dict(zip(c.manifest['joint_names']['right'],cmd[7:],strict=True)))['right_index_tip'][:3,3]
        camera=c.cameras[109]; C=(tip@c.stage-camera[:3,3])@camera[:3,:3]
        analytic=H@np.r_[point(*angles),1.]; closure=float(abs(C-analytic).max()); maximum=max(maximum,closure)
        err=float(np.linalg.norm(c.uv(C[None])[0]-pixel)) if C[2]>.01 else None
        triple=np.r_[pair,angles[None]]; d=triple-c.aligned[107:110,1,7:9]
        checks=dict(actual_position=bool(((triple>=c.limits[:2,0]-1e-8)&(triple<=c.limits[:2,1]+1e-8)).all()),
            component=bool((abs(d)<=.25+1e-7).all()),
            independent_original_position=bool((abs(d)<=.25+1e-8).all()),
            adjacent=bool((abs(np.diff(d,axis=0))<=.05+1e-7).all()),
            second=bool((abs(np.diff(d,n=2,axis=0))<=.05+1e-7).all()),
            velocity=bool((abs(np.diff(triple,axis=0))<=12*np.diff(c.ts[107:110])[:,None]+1e-7).all()),
            source=bool(C[2]>.01 and err<=cap+1e-6), URDF_binding=bool(closure<1e-8))
        record=dict(pair_ids=pair_ids,pair=pair,angles_rad=angles,command=cmd,actual_URDF_camera_point=C,
            source_error_px=err,ORIGINAL_source_cap_px=cap,URDF_analytic_camera_closure_m=closure,
            triple_corrections=d,checks=checks,passed=all(checks.values()),prefix_and_future_viability_unassessed=True)
        g.journal(j,dict(kind='AFTER_independent_new_URDF_source_triple',record=record));witness_checks+=1
        (verified if record['passed'] else rejected).append(record)
    if not row['classification'].startswith('unresolved'):continue
    box=np.c_[row['box']['joint_lower'],row['box']['joint_upper']]; center=box.mean(axis=1)
    C=H@np.r_[point(*center),1.]; B=np.array([[c.K[0,0],0,c.K[0,2]-pixel[0]],[0,c.K[1,1],c.K[1,2]-pixel[1]]])
    residual=B@C
    u=residual/np.linalg.norm(residual)
    # Outward original IMAGE tolerance belongs in the dual radius. The
    # additional 1e-8 m binding guard is a necessary-bound safety margin.
    n=B.T@u-(cap+1e-6)*np.array([0.,0.,1.]);n=n/np.linalg.norm(n)
    planar=H[:,:2].T@n;constant=float(n@H[:,2])
    g.journal(j,dict(kind='BEFORE_new_unresolved_box_support',pair_ids=pair_ids,box=box,unit_image_dual=u,unit_camera_plane=n))
    minimum, candidates=support(planar,box[0],box[1]); minval=minimum['projection']+constant
    rng=np.random.default_rng(109000+pair_ids[0]*100+pair_ids[1]); sampleangles=rng.uniform(box[:,0],box[:,1],size=(32,2)); samplevals=[];closures=[]
    for a,b in sampleangles:
        cmd=commands[109,1].copy();cmd[7:9]=[a,b]
        tip=kin.poses(cmd[:7],dict(zip(c.manifest['joint_names']['right'],cmd[7:],strict=True)))['right_index_tip'][:3,3]
        source=(tip@c.stage-camera[:3,3])@camera[:3,:3]
        samplevals.append(float(n@source));closures.append(float(abs(source-H@np.r_[point(a,b),1.]).max()))
    record=dict(pair_ids=pair_ids,box=box,source_cone_radius_with_original_tolerance_px=cap+1e-6,
        unit_image_dual=u,unit_camera_plane=n,planar_normal=planar,constant=constant,
        exact_numerical_workspace_minimum=minimum,all_four_boundary_critical_points=candidates,
        minimum_camera_plane_m=minval,binding_guard_m=1e-8,
        classification='numerically_separated_from_original_source_cone' if minval>1e-8 else 'unresolved_no_separating_plane_from_this_dual',
        independent_URDF_sample_angles=sampleangles,independent_camera_plane_samples_m=samplevals,
        independent_binding_closures_m=closures,formal_directed_rounding_proof=False)
    g.journal(j,dict(kind='AFTER_new_unresolved_box_support',record=record));support_rows.append(record)
    assert min(samplevals)>=minval-1e-8 and max(closures)<1e-8
j.close()
(O/'verified-new-index-triples.json').write_text(json.dumps(g.plain(dict(verified=verified,rejected=rejected,support_rows=support_rows)),allow_nan=False,indent=2)+'\n')
out=dict(schema='v2d.independent-new-index-temporal-source-boxes.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    producer_sha256=sha(__file__),input_sha256=sha(path),independent_new_URDF_source_triples=witness_checks,
    accepted_triples=len(verified),rejected_triples=len(rejected),maximum_URDF_analytic_closure_m=maximum,
    additional_URDF_support_samples=32*len(support_rows),new_support_boxes=len(support_rows),
    new_support_classifications={k:sum(v['classification']==k for v in support_rows) for k in set(v['classification'] for v in support_rows)},
    source_chart_solves_not_repeated=True,old_graph_forward_pairs_not_reconstructed=True,
    local_three_frame_compatibility_NOT_prefix_future_or_whole_clock=True,
    numerical_planes_not_formal_directed_rounding_proof=True,
    current_verified_reference_and_source_authority_unchanged=True,new_native_position_checks=0,
    new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,whole_reference_or_task_qualified=False,
    all_outputs_sha256={p.name:sha(p) for p in O.iterdir() if p.is_file() and p.name not in ['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(g.plain(out)))
