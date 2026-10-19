"""NEW exact-form cone preimages; reuse 12 unresolved boxes without splitting.

Polynomial arithmetic is floating point. Neither a numerical Bernstein range
nor a source-only algebraic result is a reference, temporal or native proof.
"""
from pathlib import Path
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-index-source-polynomial-v1';sys.path.insert(0,str(S/'compute'))
import source_orientation_context_v1 as c
import thumb_conic_geometry_v1 as g
from planar_2R_source_polynomial_v1 import source_polynomial,source_box_chart,bernstein_range
from scipy.spatial.transform import Rotation
np=c.np;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic();rng=np.random.default_rng(19191021)
j=(O/'all-new-polynomial-controls-proposals-and-domain-records.jsonl').open('w');controls=[]
def require(name,passed,evidence):
    g.journal(j,dict(kind='new_polynomial_analytic_control',name=name,passed=bool(passed),evidence=evidence));controls.append(dict(name=name,passed=bool(passed)));assert passed,name
H=np.array([[.9,-.2,.04],[.1,1.1,-.08],[.12,.03,.6]]);K=np.array([[800.,0,320.],[0,810.,240.],[0,0,1.]]);pixel=np.array([370.,205.]);radius=25.
polynomial=source_polynomial(H,K,pixel,radius);max_camera=0.;max_source=0.;samples=[]
for a,b in rng.uniform([-2.,0.],[2.,2.5],size=(256,2)):
    u,v=np.tan([a,b]/2);D=np.polynomial.polynomial.polyval2d(u,v,polynomial['denominator']);C=np.array([np.polynomial.polynomial.polyval2d(u,v,p) for p in polynomial['camera_numerators']])/D
    z=.0458*np.exp(1j*a)+.035*np.exp(1j*(a+b));exact=H@np.array([z.real,z.imag,1.]);closure=float(abs(C-exact).max());max_camera=max(max_camera,closure)
    residual=np.array([K[0,0]*exact[0]+(K[0,2]-pixel[0])*exact[2],K[1,1]*exact[1]+(K[1,2]-pixel[1])*exact[2]])
    direct=float(residual@residual-radius**2*exact[2]**2)
    value=np.polynomial.polynomial.polyval2d(u,v,polynomial['normalized_source_polynomial'])*polynomial['source_scale']/D**2
    err=float(abs(value-direct)/max(1.,abs(direct)));max_source=max(max_source,err);samples.append(dict(angles=[a,b],camera_closure_m=closure,relative_cone_identity_error=err))
require('complex_exponential_camera_numerator_identity',max_camera<1e-12,dict(maximum_m=max_camera,samples=samples))
require('source_disk_quartic_identity',max_source<1e-10,dict(maximum_relative_error=max_source,bidegree=polynomial['source_bidegree']))
p=np.array([[0.,0.,1.],[0.,2.,0.],[1.,0.,0.]]);r=bernstein_range(p,[[0.,1.],[0.,1.]])
require('known_bivariate_square_range',r['lower']<=0 and r['upper']>=4 and abs(r['upper']-4)<1e-8,r)
negative=source_polynomial(np.diag([1.,1.,-1.]),np.eye(3),[0,0],100.)
negativechart=source_box_chart(negative,[[-.2,.2],[.2,.5]])
require('positive_depth_is_separate_from_disk',negativechart['classification']=='numerically_empty_positive_depth_preimage',negativechart)
radial=source_polynomial(np.eye(3),np.eye(3),[0,0],.0818)
radialerrors=[]
for a in np.linspace(-1.,1.,17):
    u=np.tan(a/2);D=np.polynomial.polynomial.polyval2d(u,0.,radial['denominator']);F=np.polynomial.polynomial.polyval2d(u,0.,radial['source_polynomial'])
    radialerrors.append(abs(F/D**2-((.0458+.035)**2-.0818**2)))
require('straight_distal_boundary_preserved',max(radialerrors)<1e-15 and source_box_chart(radial,[[-.1,.1],[0.,.2]])['distal_zero_boundary_present'],dict(errors=radialerrors))
require('half_angle_pole_not_silently_discarded',source_box_chart(radial,[[3.1,3.2],[.1,.3]])['classification']=='unresolved_half_angle_seam_or_pole',dict(bounds=[[3.1,3.2],[.1,.3]]))
fixed=np.array([.3,.6]);uv=np.tan(fixed/2);value=float(np.polynomial.polynomial.polyval2d(*uv,radial['normalized_source_polynomial']));rr=source_box_chart(radial,np.c_[fixed,fixed])['source_range']
require('zero_width_chart_encloses_exact_value',rr['lower']<=value<=rr['upper'],dict(value=value,range=rr))
bad=np.eye(3);bad[0,1]=.1;rejected=False
try:source_polynomial(np.eye(3),bad,[0,0],1.)
except ValueError:rejected=True
require('unknown_skew_camera_contract_rejected',rejected,dict(skew=.1))

with np.load(S/'e037-strict-source-wrist-mode-v1/UNQUALIFIED-strict-source-root-proposal.npz') as z:commands=z['robot_qpos'].copy();weights=z['weights'].copy()
source_rows={v['frame']:v for v in json.loads((S/'e037-thumb-ray-independent-validation-v1/report.json').read_text())['source_rows'] if v['hand']=='right'}
polynomials={};records=[]
for frame in np.flatnonzero(weights>1e-14):
    frame=int(frame);cmd=commands[frame,1];camera=c.cameras[frame];pixel=c.uv((c.human[frame,1,[8]]-camera[:3,3])@camera[:3,:3])[0];cap=source_rows[frame]['source_before_after_px'][0][2]
    A=camera[:3,:3].T@c.stage.T;R=Rotation.from_quat(cmd[3:7]).as_matrix();H=np.c_[A@R@np.eye(3)[:,:2],A@(cmd[:3]+R@np.array([.0777,-.0016,.0285]))-camera[:3,:3].T@camera[:3,3]]
    lo,hi=c.bounds(frame);box=np.c_[lo[:2]-1e-8,hi[:2]+1e-8]
    g.journal(j,dict(kind='BEFORE_NEW_actual_index_source_polynomial',frame=frame,H=H,K=c.K,pixel=pixel,ORIGINAL_cap_px=cap,ORIGINAL_image_tolerance_px=1e-6))
    poly=source_polynomial(H,c.K,pixel,cap+1e-6);chart=source_box_chart(poly,box)
    record=dict(frame=frame,H=H,K=c.K,pixel=pixel,ORIGINAL_cap_px=cap,ORIGINAL_image_tolerance_px=1e-6,polynomial=poly,full_original_position_box_chart=chart,
        frame_direct_flag=bool(c.flags['hand_directly_observed'][frame,1]),frame_metric_flag=bool(c.flags['hand_metric_valid'][frame,1]),flags_NOT_measured_robot_joint_orientation_or_contact=True)
    g.journal(j,dict(kind='AFTER_NEW_actual_index_source_polynomial',record=record));records.append(record);polynomials[frame]=poly

old=S/'e037-index-source-temporal-cone-branch-v1/processed-and-unresolved-domains.json';oldreport=json.loads((old.parent/'report.json').read_text());assert sha(old)==oldreport['all_outputs_sha256'][old.name]
unresolved=json.loads(old.read_text())['unresolved'];assert len(unresolved)==12;regions=[];counts={}
for region in unresolved:
    box=np.array(region['box']);charts=[]
    g.journal(j,dict(kind='BEFORE_NEW_polynomial_chart_of_saved_outer_domain',saved_node_id=region['id'],saved_domain_sha256=sha(old),box=box))
    for frame,poly in polynomials.items():
        chart=source_box_chart(poly,box[frame]);charts.append(dict(frame=frame,chart=chart));counts[chart['classification']]=counts.get(chart['classification'],0)+1
    separated=[v['frame'] for v in charts if v['chart']['classification'].startswith('numerically_empty')]
    record=dict(saved_node_id=region['id'],box=box,source_charts=charts,numerically_source_empty_frames=separated,
        classification='numerically_source_empty_saved_outer_region' if separated else 'unresolved_coupled_source_and_temporal_region')
    regions.append(record);g.journal(j,dict(kind='AFTER_NEW_polynomial_chart_of_saved_outer_domain',record=record))
j.close();(O/'actual-index-polynomials-and-saved-region-charts.json').write_text(json.dumps(g.plain(dict(polynomials=records,regions=regions)),allow_nan=False,indent=2)+'\n')
out=dict(schema='v2d.actual-index-source-polynomial.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),new_analytic_controls=controls,
    source_frames=len(records),new_bidegree_4_4_source_preimages=len(records),saved_outer_regions_reused_without_split=len(regions),source_region_charts=sum(len(v['source_charts']) for v in regions),chart_classifications=counts,
    numerically_empty_saved_regions=sum(bool(v['numerically_source_empty_frames']) for v in regions),remaining_unresolved_regions=sum(not v['numerically_source_empty_frames'] for v in regions),
    no_old_Taylor_LP_partition_conic_or_source_fit_reexecution=True,no_new_distal_or_point_bank=True,positive_depth_half_angle_poles_singular_distal_and_unresolved_scopes_retained=True,
    floating_polynomial_Bernstein_and_tan_bounds_NOT_formal_directed_rounding_proofs=True,
    source_authority_current_verified_reference_and_middle_component_unchanged=True,
    new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,no_full_clock_candidate_qualified=True,
    all_outputs_sha256={p.name:sha(p) for p in O.iterdir() if p.is_file() and p.name not in ['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(g.plain(out)))
