"""Independent URDF, rational identity and saved Bernstein basis checks.

No polynomial producer, source chart solver, old LP or partition executes.
Saved Bernstein coefficients are evaluated in their OWN basis, independently
of the saved power coefficients. Numerical checks are not formal certificates.
"""
from pathlib import Path
import datetime,hashlib,json,math,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-index-source-polynomial-independent-v1';sys.path.insert(0,str(S/'compute'))
import source_orientation_context_v1 as c
import thumb_conic_geometry_v1 as g
from v2d_dex3_flashchord.kinematics import URDFKinematics
np=c.np;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic()
P=S/'e037-index-source-polynomial-v2';path=P/'actual-index-polynomials-and-saved-region-charts.json';report=json.loads((P/'report.json').read_text());assert sha(path)==report['all_outputs_sha256'][path.name]
saved=json.loads(path.read_text());records={v['frame']:v for v in saved['polynomials']}
with np.load(S/'e037-strict-source-wrist-mode-v1/UNQUALIFIED-strict-source-root-proposal.npz') as z:commands=z['robot_qpos'].copy()
kin=URDFKinematics(c.F/'robot/urdfs/g1/dex3_right.urdf');rng=np.random.default_rng(19191022)
j=(O/'all-independent-URDF-polynomial-and-Bernstein-checks.jsonl').open('w')
checks=0;camera_max=0.;source_max=0.;depth_max=0.
for frame,record in records.items():
    poly=record['polynomial'];box=np.array(record['full_original_position_box_chart']['angle_bounds'])
    for angles in rng.uniform(box[:,0],box[:,1],size=(16,2)):
        cmd=commands[frame,1].copy();cmd[7:9]=angles
        g.journal(j,dict(kind='BEFORE_independent_new_URDF_polynomial_binding',frame=frame,command=cmd))
        tip=kin.poses(cmd[:7],dict(zip(c.manifest['joint_names']['right'],cmd[7:],strict=True)))['right_index_tip'][:3,3]
        camera=c.cameras[frame];C=(tip@c.stage-camera[:3,3])@camera[:3,:3];u,v=np.tan(angles/2);D=(1+u*u)*(1+v*v)
        prediction=np.array([np.polynomial.polynomial.polyval2d(u,v,p) for p in poly['camera_numerators']])/D
        closure=float(abs(C-prediction).max());camera_max=max(camera_max,closure)
        K=np.array(record['K']);pixel=np.array(record['pixel']);radius=record['ORIGINAL_cap_px']+record['ORIGINAL_image_tolerance_px']
        e=np.array([K[0,0]*C[0]+(K[0,2]-pixel[0])*C[2],K[1,1]*C[1]+(K[1,2]-pixel[1])*C[2]])
        Fdirect=float((e@e-radius**2*C[2]**2)*D**2/poly['source_scale']);F=float(np.polynomial.polynomial.polyval2d(u,v,poly['normalized_source_polynomial']))
        Gdirect=float((C[2]-.01)*D/poly['depth_scale']);G=float(np.polynomial.polynomial.polyval2d(u,v,poly['normalized_depth_polynomial']))
        source_error=abs(F-Fdirect);depth_error=abs(G-Gdirect);source_max=max(source_max,source_error);depth_max=max(depth_max,depth_error)
        passed=bool(closure<1e-8 and source_error<1e-10 and depth_error<1e-10)
        g.journal(j,dict(kind='AFTER_independent_new_URDF_polynomial_binding',frame=frame,angles=angles,actual_camera_point=C,camera_closure_m=closure,
            normalized_source_identity_error=source_error,normalized_depth_identity_error=depth_error,passed=passed));assert passed;checks+=1

def evaluate_bernstein(coefs,t,s):
    coefs=np.array(coefs);n,m=np.array(coefs.shape)-1
    return math.fsum(float(coefs[i,k])*math.comb(n,i)*t**i*(1-t)**(n-i)*math.comb(m,k)*s**k*(1-s)**(m-k) for i in range(n+1) for k in range(m+1))

basischecks=0;maximum_basis_error=0.;regionrecords=[]
for region in saved['regions']:
    region_passed=True;separated=[];all_checks=[]
    for item in region['source_charts']:
        frame=item['frame'];chart=item['chart'];poly=records[frame]['polynomial'];half=np.array(chart['half_angle_bounds'])
        for key,polynomial_key in [('source_range','normalized_source_polynomial'),('depth_range','normalized_depth_polynomial')]:
            range_record=chart[key];coeff=np.array(range_record['bernstein_coefficients']);n,m=np.array(coeff.shape)-1;errors=[];in_range=[]
            for t in np.linspace(0,1,n+1):
                for s in np.linspace(0,1,m+1):
                    uv=half[:,0]+(half[:,1]-half[:,0])*[t,s]
                    power=float(np.polynomial.polynomial.polyval2d(*uv,poly[polynomial_key]));bernstein=evaluate_bernstein(coeff,t,s)
                    difference=abs(power-bernstein);maximum_basis_error=max(maximum_basis_error,difference);errors.append(difference)
                    in_range.append(bool(range_record['lower']<=power<=range_record['upper']));basischecks+=1
            passed=bool(max(errors)<=range_record['numerical_outward_padding'] and all(in_range));region_passed &= passed
            check=dict(frame=frame,range_kind=key,maximum_independent_basis_error=max(errors),saved_outward_padding=range_record['numerical_outward_padding'],passed=passed)
            all_checks.append(check);g.journal(j,dict(kind='independent_saved_Bernstein_basis_and_range_check',saved_node_id=region['saved_node_id'],record=check));assert passed
        if chart['classification'].startswith('numerically_empty'):
            separated.append(dict(frame=frame,classification=chart['classification'],source_lower=chart['source_range']['lower'],depth_upper=chart['depth_range']['upper']))
    assert [v['frame'] for v in separated]==region['numerically_source_empty_frames']
    regionrecords.append(dict(saved_node_id=region['saved_node_id'],all_independent_basis_checks_passed=region_passed,numerically_source_empty_frames=separated,
        classification='independently_checked_numerical_source_exclusion' if separated else 'still_unresolved_source_temporal_region',
        mathematical_float_model_not_source_accuracy_or_formal_directed_rounding_proof=True))
j.close();(O/'independently-checked-source-region-records.json').write_text(json.dumps(g.plain(regionrecords),allow_nan=False,indent=2)+'\n')
out=dict(schema='v2d.independent-index-source-polynomial.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),
    independent_actual_URDF_camera_source_polynomial_states=checks,maximum_camera_closure_m=camera_max,
    maximum_normalized_source_polynomial_identity_error=source_max,maximum_normalized_depth_polynomial_identity_error=depth_max,
    independent_tensor_Bernstein_basis_values=basischecks,maximum_independent_basis_error=maximum_basis_error,
    all_samples_within_saved_padding_and_enclosures=True,independently_checked_numerically_excluded_regions=sum(bool(v['numerically_source_empty_frames']) for v in regionrecords),
    remaining_unresolved_regions=sum(not v['numerically_source_empty_frames'] for v in regionrecords),source_polynomial_producer_and_Bernstein_conversion_not_reexecuted=True,
    no_source_accuracy_formal_global_infeasibility_or_temporal_certificate_claimed=True,
    current_verified_reference_middle_component_source_authority_and_parameters_unchanged=True,
    new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,no_full_clock_candidate_qualified=True,
    all_outputs_sha256={p.name:sha(p) for p in O.iterdir() if p.is_file() and p.name not in ['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(g.plain(out)))
