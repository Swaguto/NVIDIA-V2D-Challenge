from thumb_ray_context_v1 import *
import datetime,subprocess,xml.etree.ElementTree as ET
from v2d_dex3_flashchord.planar_thumb_ik import PlanarThumbChain,thumb_point,thumb_point_branches
from scipy.spatial.transform import Rotation
O=S/'e037-thumb-ray-library-verification-v1';start=time.monotonic()
test=R/'scripts/dex3_training/tests/test_planar_thumb_ik.py';library=R/'scripts/dex3_training/v2d_dex3_flashchord/planar_thumb_ik.py'
p=subprocess.run([sys.executable,'-m','pytest','-q',str(test)],cwd=R,capture_output=True,text=True);(O/'pytest.log').write_text(p.stdout+p.stderr)
chain=PlanarThumbChain((.0255,0,0),(-.0025,.0193),.0458,.035)
root=ET.parse(F/'robot/urdfs/g1/dex3_right.urdf').getroot();joints={j.attrib['name']:j for j in root.findall('joint')}
for name,axis,origin in [('right_hand_thumb_0_joint',[0,1,0],[.0255,0,0]),('right_hand_thumb_1_joint',[0,0,1],[-.0025,.0193,0]),('right_hand_thumb_2_joint',[0,0,1],[0,.0458,0])]:
 j=joints[name];np.testing.assert_array_equal(np.fromstring(j.find('axis').attrib['xyz'],sep=' '),axis);np.testing.assert_allclose(np.fromstring(j.find('origin').attrib['xyz'],sep=' '),origin,atol=1e-15);assert j.find('origin').attrib['rpy']=='0 0 0'
np.testing.assert_allclose(np.fromstring(joints['right_thumb_tip_joint'].find('origin').attrib['xyz'],sep=' '),[0,.035,0],atol=1e-15)
rows=[]
for frame in [0,4,110,111,113,116,119,121,150]:
 set_state(frame);Rroot=Rotation.from_quat(q[frame,1,3:7]).as_matrix();actual=np.einsum('i,ij->j',data.xpos[tipids[1]]-q[frame,1,:3],Rroot,optimize=False);analytic=thumb_point(chain,q[frame,1,11:]);lo,hi=bounds(frame);branches=thumb_point_branches(chain,actual,np.c_[lo[4:],hi[4:]]);closure=float(np.max(abs(analytic-actual)));recovered=any(np.max(abs(np.array(b)-q[frame,1,11:]))<1e-6 for b in branches.joints_rad);rows.append(dict(frame=frame,analytic_native_closure_m=closure,actual_angles_recovered=recovered,branches=[list(b) for b in branches.joints_rad]))
report=dict(schema='v2d.analytic-thumb-library-binding.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),context_sha256=sha(S/'compute/thumb_ray_context_v1.py'),library_sha256=sha(library),test_sha256=sha(test),pytest_exit_code=p.returncode,pytest_output=p.stdout.strip(),URDF_axis_origin_length_contract_verified=True,native_binding_rows=rows,library_and_binding_passed=bool(p.returncode==0 and all(r['analytic_native_closure_m']<1e-8 and r['actual_angles_recovered'] for r in rows)),combined_freeze_unchanged=True,source_truth_or_collision_certified=False,no_native_compile_or_physics_steps=True,new_GPU_jobs=0,new_PPO_steps=0,wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
