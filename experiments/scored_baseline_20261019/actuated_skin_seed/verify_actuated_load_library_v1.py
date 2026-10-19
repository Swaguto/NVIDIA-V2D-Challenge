"""New actuator-aware mechanics tests and read-only native cap binding."""
from pathlib import Path
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');sys.path.insert(0,str(R/'scripts/dex3_training'))
import mujoco,numpy as np,pytest
from audit_hand_reference_contacts import canonical_id
start=time.monotonic();O=S/'actuated-load-library-verification-v1';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
paths=[R/'scripts/dex3_training/tests'/n for n in ['test_actuated_contact_load.py','test_contact_load_feasibility.py','test_wrench_capacity.py']]
exit_code=pytest.main(['-q',*(str(p) for p in paths),'--tb=short']);assert exit_code==0
P=S/'e037-motion-constrained-wrist-native-v1';manifest=json.loads((P/'frozen/manifest.json').read_text());model_path=P/'native-model.mjb';model=mujoco.MjModel.from_binary_path(str(model_path));names=[model.joint(i).name for i in range(model.njnt)]
rows=[]
for name in manifest['joint_names']['right']:
 j=canonical_id(names,name);assert model.jnt_actfrclimited[j] and model.jnt_actgravcomp[j];np.testing.assert_allclose(model.jnt_actfrcrange[j],[-.76,.76],atol=2e-8,rtol=0)
 rows.append(dict(joint=name,native_joint_id=j,native_dof_id=int(model.jnt_dofadr[j]),native_effort_range_Nm=model.jnt_actfrcrange[j].tolist(),native_actuator_gravity_compensation_inside_clamp=bool(model.jnt_actgravcomp[j]),native_stiffness=float(model.jnt_stiffness[j]),native_frictionloss_Nm=float(model.dof_frictionloss[model.jnt_dofadr[j]])))
report=dict(schema='v2d.actuated-contact-load-library-verification.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),library_sha256=sha(R/'scripts/dex3_training/v2d_dex3_flashchord/contact_load_feasibility.py'),test_sha256={p.name:sha(p) for p in paths},pytest_exit_code=int(exit_code),native_model_sha256=sha(model_path),native_joint_effort_binding=rows,primary_reference='https://mujoco.readthedocs.io/en/3.4.0/XMLreference.html#body-joint-actuatorgravcomp',native_values_read_not_recompiled=True,no_native_forward_or_old_saved_contact_compatibility_or_source_job_repeated=True,source_or_commands_not_changed=True,new_GPU_work=False,new_PPO_steps=0,physical_task_accepted=False,wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
