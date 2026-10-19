"""New common-root facet LP mechanics tests; no old job repeats."""
from pathlib import Path
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');sys.path.insert(0,str(R/'scripts/dex3_training'))
import pytest
O=S/'contact-translation-library-verification-v1';start=time.monotonic();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();p=R/'scripts/dex3_training/v2d_dex3_flashchord/contact_translation.py';test=R/'scripts/dex3_training/tests/test_contact_translation.py';code=pytest.main(['-q',str(test),'--tb=short']);assert code==0
report=dict(schema='v2d.common-contact-translation-verification.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),library_sha256=sha(p),test_sha256=sha(test),pytest_exit_code=int(code),no_native_compile_or_physics_steps=True,prior_jobs_not_repeated=True,new_GPU_jobs=0,new_PPO_steps=0,training_approved=False,physical_task_accepted=False,wall_seconds=time.monotonic()-start);(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
