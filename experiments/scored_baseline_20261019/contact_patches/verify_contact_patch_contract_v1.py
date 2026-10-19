"""New patch-contract tests and signed native gap-gradient mechanics controls."""
from pathlib import Path
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');sys.path.insert(0,str(R/'scripts/dex3_training'))
import mujoco,numpy as np,pytest
O=S/'contact-patch-contract-verification-v1';start=time.monotonic();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();test=R/'scripts/dex3_training/tests/test_contact_patches.py';code=R/'scripts/dex3_training/v2d_dex3_flashchord/contact_patches.py';exit_code=pytest.main(['-q',str(test),'--tb=short']);assert exit_code==0
# New two-box analytic model only; no asset recompilation or old job rerun.
model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="moving"><joint type="slide" axis="1 0 0"/><geom type="box" size=".1 .1 .1"/></body><geom type="box" pos=".5 0 0" size=".1 .1 .1"/></worldbody></mujoco>');data=mujoco.MjData(model);moving=int(np.flatnonzero(model.geom_bodyid==1)[0]);fixed=int(np.flatnonzero(model.geom_bodyid==0)[0]);rows=[]
for q,expected in [(0.,.3),(.35,-.05)]:
 data.qpos[0]=q;mujoco.mj_fwdPosition(model,data);segment=np.zeros(6);gap=float(mujoco.mj_geomDistance(model,data,moving,fixed,1.,segment));assert abs(gap-expected)<1e-12
 normal=(segment[3:]-segment[:3])/gap;np.testing.assert_allclose(normal,[1,0,0],atol=1e-10);jp=np.zeros((3,model.nv));mujoco.mj_jac(model,data,jp,None,segment[:3],1);gradient=-np.einsum('i,ij->j',normal,jp,optimize=False);distances=[]
 for delta in [-1e-6,1e-6]:data.qpos[0]=q+delta;mujoco.mj_fwdPosition(model,data);distances.append(mujoco.mj_geomDistance(model,data,moving,fixed,1.,None))
 fd=(distances[1]-distances[0])/2e-6;assert abs(fd-gradient[0])<1e-8;rows.append(dict(qpos=q,signed_gap_m=gap,gap_gradient=float(gradient[0]),central_difference_gradient=float(fd)))
report=dict(schema='v2d.contact-patch-contract-verification.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),library_sha256=sha(code),test_sha256=sha(test),pytest_exit_code=int(exit_code),native_signed_distance_controls=rows,compiled_only_new_analytic_box_control=True,old_native_source_or_contact_jobs_not_repeated=True,physics_stepped=False,new_GPU_work=False,new_PPO_steps=0,training_approved=False,physical_task_accepted=False,wall_seconds=time.monotonic()-start);(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
