"""NEW full-clock hand-self scope with cached model; tests and native audit."""
from pathlib import Path
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');sys.path.insert(0,str(R/'scripts/dex3_training'))
import pytest
from audit_hand_self_contacts import audit
O=S/'e037-cached-hand-self-contacts-v1';start=time.monotonic();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();plan_path=S/'compute/cached-hand-self-contact-CPU-plan-v1.json';plan=json.loads(plan_path.read_text());code=R/'scripts/dex3_training/audit_hand_self_contacts.py';lib=R/'scripts/dex3_training/v2d_dex3_flashchord/self_contact.py';test=R/'scripts/dex3_training/tests/test_self_contact.py';result=pytest.main(['-q',str(test),'--tb=short']);assert result==0
P=S/'e037-motion-constrained-wrist-native-v1';report=audit(P/'frozen',P/'native-model.mjb',plan['native_model_sha256']);report.update(wrapper_producer_sha256=sha(Path(__file__)),plan_sha256=sha(plan_path),library_sha256=sha(lib),test_sha256=sha(test),pytest_exit_code=int(result),source_inputs_sha256={str(p):sha(p) for p in [P/'frozen/manifest.json',P/'frozen/targets.npz',P/'frozen/source-confidence.npz',P/'native-model.mjb']},new_GPU_jobs=0,new_PPO_steps=0);(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['reference_knots','summary_by_hand','summary_by_pair','hand_self_contact_gate_passed','maximum_native_FK_error','wall_seconds']},indent=2),flush=True)
