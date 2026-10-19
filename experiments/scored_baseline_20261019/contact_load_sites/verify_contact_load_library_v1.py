"""New focused mechanics tests and cached-contact compatibility; no native job."""
from pathlib import Path
import datetime, hashlib, json, sys, time
S=Path(__file__).resolve().parents[1]
R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path.insert(0,str(R/'scripts/dex3_training'))
import pytest
from v2d_dex3_flashchord.contact_load_feasibility import ContactGeometry, outer_contact_load_feasibility
start=time.monotonic()
O=S/'contact-load-library-verification-v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
meta_path=S/'compute/palm-wrench-library-compatibility-inputs-v1.json'
meta=json.loads(meta_path.read_text())
source=Path(meta['source_report'])
assert sha(source)==meta['source_report_sha256']
old=json.loads(source.read_text())
code=R/'scripts/dex3_training/v2d_dex3_flashchord/contact_load_feasibility.py'
test=R/'scripts/dex3_training/tests/test_contact_load_feasibility.py'
exit_code=pytest.main(['-q',str(test),str(R/'scripts/dex3_training/tests/test_wrench_capacity.py'),'--tb=short'])
assert exit_code==0,exit_code
rows=[]
for row in old['rows']:
    contacts=[ContactGeometry(c['point'],c['frame'],c['dim'],c['friction']) for c in row['contacts']]
    outputs={}
    for label,field,control in [
        ('gravity','gravity_load_wrench_world','gravity_outer_friction_feasibility'),
        ('estimated_motion','estimated_reference_motion_load_wrench_world','estimated_motion_outer_friction_feasibility')]:
        result=outer_contact_load_feasibility(contacts,meta['object_COM_native_world_m'],row[field])
        assert result['feasible_in_outer_model']==row[control]['feasible']
        outputs[label]=result
    rows.append(dict(source_label=row['source_label'],contacts=len(contacts),results=outputs))
report=dict(schema='v2d.contact-load-library-verification.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),library_sha256=sha(code),test_sha256=sha(test),source_report_sha256=sha(source),compatibility_metadata_sha256=sha(meta_path),pytest_exit_code=int(exit_code),cached_contact_load_cases=len(rows)*2,all_six_saved_infeasibility_results_match=True,rows=rows,no_native_recompile_or_forward_or_source_job_rerun=True,new_GPU_work=False,new_PPO_steps=0,no_candidate_promoted=True,wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
