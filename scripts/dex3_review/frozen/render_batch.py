"""Render approved preview configuration with an explicit compute deadline."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

R=Path('/workspace/hand-continuity-repair/dex3-isaac-review')
EPISODES=[11,12,2,13,18,21,23,24,31,40,41,42]
deadline=datetime.datetime.fromisoformat('2026-10-15T05:45:00+00:00').timestamp()
while not (R/'pilot-review-approved.json').is_file():
    if time.time()>deadline:raise RuntimeError('Preview deadline reached before pilot approval')
    time.sleep(10)
approval=json.loads((R/'pilot-review-approved.json').read_text())
PY=Path(approval.get('python',str(R/'runtime6/bin/python')))
LAB=Path(approval.get('lab',str(R/'IsaacLab3')))
env=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',OMNI_KIT_ALLOW_ROOT='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
env.pop('PYTHONPATH',None)
status={}
for ep in EPISODES:
    label=f'e{ep:03d}'
    name=approval['pilot_sim_name'] if ep==11 else 'sim-v1'
    out=R/'episodes'/label/name
    if not (out/'complete.json').is_file():
        print('ISAAC_START',label,flush=True)
        cmd=[str(PY),str(R/'replay_continuity_dex3_sim6.py'),'--source',str(R/'retarget-code/robotic_grounding'),'--bundle',str(R/'episodes'/label/'bundle'),'--retarget',str(R/'episodes'/label/'retarget'),'--out',str(out),'--width','1280','--height','800','--headless','--enable_cameras','--device','cpu','--rendering_mode','performance']
        with (out.parent/(name+'.log')).open('w') as log:
            result=subprocess.run(cmd,cwd=LAB,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=min(1500,max(1,deadline-time.time())))
        if result.returncode or not (out/'complete.json').is_file():
            raise RuntimeError('Isaac render failed: '+label)
    print('ISAAC_COMPLETE',label,flush=True)
    with (out.parent/'package.log').open('w') as log:
        result=subprocess.run(['/opt/conda/bin/python',str(R/'package_review.py'),str(ep),'--sim-name',name],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='2'),timeout=min(300,max(1,deadline-time.time())))
    status[label]={'render_complete':True,'package_exit_code':result.returncode,'sim_dir':str(out)}
    (R/'render-status.json').write_text(json.dumps(status,indent=2))
    if result.returncode:raise RuntimeError('Review package failed: '+label)
    print('ISAAC_REVIEW_READY',label,flush=True)
print('ISAAC_BATCH_COMPLETE',len(status),flush=True)
