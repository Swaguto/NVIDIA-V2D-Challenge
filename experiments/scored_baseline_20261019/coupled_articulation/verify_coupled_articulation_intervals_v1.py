"""NEW numerical enclosure controls and actual distal ancestry binding."""
from pathlib import Path
import datetime,hashlib,json,math,sys,time,xml.etree.ElementTree as ET
S=Path(__file__).resolve().parents[1];O=S/'e037-coupled-articulation-binding-v1';start=time.monotonic();sys.path.insert(0,str(S/'compute'))
import coupled_articulation_intervals_v1 as b
import source_orientation_context_v1 as c
from scipy.spatial.transform import Rotation
np=c.np;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();controls=[]
def record(name,passed,**details):
 controls.append(dict(name=name,passed=bool(passed),**details));(O/'all-controls.json').write_text(json.dumps(controls,indent=2)+'\n');assert passed,name
def contains(box,value):return bool((value>=box[...,0]-1e-11).all() and (value<=box[...,1]+1e-11).all())
record('full-turn_and_stationary-trig_extrema',contains(b.sine(np.array([-.4,2*math.pi])),np.array(-1.)) and contains(b.cosine(np.array([-.2,.2])),np.array(1.)))
record('zero-containing_square_nonnegative_enclosure',contains(b.square(np.array([-.2,.3])),np.array(0.)) and b.square(np.array([-.2,.3]))[0]<=0)
record('source-disk_separation',abs(b.disk_lower_bound(np.array([[3.,5.],[4.,6.]]),np.zeros(2))-5)<1e-14)
try:b.divide(b.exact(1.),np.array([-1.,1.]))
except ValueError:record('zero-denominator_rejected',True)
else:record('zero-denominator_contract',False)
rng=np.random.default_rng(861);maximum_native=0.;sample_records=[]
for case in range(12):
 r0=rng.uniform(-.055,.055,3);width=rng.uniform(.0001,.02,3);rb=np.c_[r0-width,r0+width];R0=Rotation.from_rotvec(rng.normal(size=3)*.04).as_matrix();q0=Rotation.from_matrix(R0).as_quat();matrix=b.rotation_matrix_box(rb);rv=b.root_rotvec_box(rb,q0);assert rv is not None
 jb=np.array([[.2,.7],[.1,.8]]);pb=b.planar_point_box(jb,.0285);tb=b.thumb_point_box([-.95,-.97],np.array([-1.1,-.6]))
 for sample in range(32):
  r=rng.uniform(rb[:,0],rb[:,1]);R=Rotation.from_rotvec(2*math.atan(np.linalg.norm(r))*r/max(np.linalg.norm(r),1e-300)).as_matrix();angles=rng.uniform(jb[:,0],jb[:,1]);a,d=angles;p=np.array([.0777+.0458*np.cos(a)+.035*np.cos(a+d),-.0016+.0458*np.sin(a)+.035*np.sin(a+d),.0285]);distal=rng.uniform(-1.1,-.6);aa,bb=-.95,-.97;x=-.0025-.0458*np.sin(bb)-.035*np.sin(bb+distal);tp=np.array([.0255+np.cos(aa)*x,.0193+.0458*np.cos(bb)+.035*np.cos(bb+distal),-np.sin(aa)*x]);assert contains(matrix,R) and contains(rv,Rotation.from_matrix(R@R0).as_rotvec()) and contains(pb,p) and contains(tb,tp)
record('independent_Rodrigues_SO3_principal_log_and_continuous_finger_enclosures',True,independent_points=384)
frame=114;original=c.q.copy();legacy=S/'e037-right-thumb-bracketed-v2/right-thumb-reference-commands.npz'
with np.load(legacy) as z:upstream=z['robot_qpos'][frame,1,11:13].copy()
root=ET.parse(c.F/'robot/urdfs/g1/dex3_right.urdf').getroot();joints={j.get('name'):j for j in root.findall('joint')};j=joints['right_hand_thumb_2_joint'];record('actual_URDF_distal_ancestry',j.find('parent').get('link')=='right_hand_thumb_1_link' and j.find('child').get('link')=='right_hand_thumb_2_link' and j.find('axis').get('xyz')=='0 0 1',parent=j.find('parent').get('link'),child=j.find('child').get('link'),origin=j.find('origin').attrib,actual_URDF_limit=j.find('limit').attrib)
lo,hi=c.bounds(frame);parent_ids=[i for i,n in enumerate(c.bn) if n.endswith('right_hand_palm_link') or n.endswith('right_hand_thumb_1_link')];poses=[];jacobians=[]
from v2d_dex3_flashchord.planar_thumb_ik import PlanarThumbChain,thumb_point
chain=PlanarThumbChain((.0255,0,0),(-.0025,.0193),.0458,.035)
for distal in [lo[6],(lo[6]+hi[6])/2,hi[6]]:
 command=original[frame,1].copy();command[11:13]=upstream;command[13]=distal;c.q[frame,1]=command;c.set_state(frame,command[7:]);R=Rotation.from_quat(command[3:7]).as_matrix();tip=command[:3]+R@thumb_point(chain,command[11:]);closure=float(abs(tip-c.data.xpos[c.tipids[1]]).max());maximum_native=max(maximum_native,closure);poses.append(np.r_[c.data.xpos[parent_ids].flatten(),c.data.xmat[parent_ids].flatten()]);J=np.zeros((3,c.model.nv));c.mujoco.mj_jac(c.model,c.data,J,None,c.data.xpos[parent_ids[-1]],parent_ids[-1]);jacobians.append(float(abs(J[:,c.rdofs[6]]).max()));sample_records.append(dict(command=command.tolist(),distal_bounds=[float(lo[6]),float(hi[6])],parent_body_names=[c.bn[i] for i in parent_ids],parent_poses=poses[-1].tolist(),parent_thumb2_jacobian_max=jacobians[-1],native_thumb_tip=c.data.xpos[c.tipids[1]].tolist(),analytic_native_tip_closure_m=closure))
c.q=original;(O/'all-native-ancestry-states.json').write_text(json.dumps(sample_records,indent=2)+'\n');record('distal_changes_tip_without_moving_palm_or_thumb1',max(abs(p-poses[0]).max() for p in poses)<1e-14 and max(jacobians)==0 and maximum_native<3e-9,maximum_tip_closure_m=maximum_native,maximum_parent_jacobian=max(jacobians))
out=dict(schema='v2d.coupled-articulation-binding.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),library_sha256=sha(S/'compute/coupled_articulation_intervals_v1.py'),input_manifest_sha256=sha(c.F/'manifest.json'),native_model_sha256=sha(c.MP),controls=controls,numerical_enclosures_not_formal_global_proof=True,distal_geometry_still_requires_fresh_native_checks=True,source_flags_observations_and_reference_unchanged=True,no_compile_physics_or_GPU=True,whole_reference_or_task_qualified=False,wall_seconds=time.monotonic()-start);(O/'report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
