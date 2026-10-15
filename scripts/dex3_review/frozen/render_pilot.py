import datetime
import json
import os
from pathlib import Path
import subprocess
import time

R=Path('/workspace/hand-continuity-repair/dex3-isaac-review')
PY=R/'runtime6/bin/python'
LAB=R/'IsaacLab3'
deadline=datetime.datetime.fromisoformat('2026-10-15T05:45:00+00:00').timestamp()
while time.time()<deadline:
    log=(R/'runtime-install.log').read_text()
    if log.rstrip().endswith('}') and '"status": "passed"' in log[-900:] and (R/'episodes/e011/retarget/quality.json').is_file():
        break
    time.sleep(10)
else:raise RuntimeError('Runtime preparation deadline')
env=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',OMNI_KIT_ALLOW_ROOT='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
env.pop('PYTHONPATH',None)
print('SIM_RUNTIME_READY',flush=True)
with (R/'runtime-ready.json').open('w') as f:json.dump({'lock':str(R/'runtime6/replay-requirements.txt'),'lab':str(LAB),'installer_check':'passed'},f)
cmd=[str(PY),str(R/'replay_continuity_dex3_sim6.py'),'--source',str(R/'retarget-code/robotic_grounding'),'--bundle',str(R/'episodes/e011/bundle'),'--retarget',str(R/'episodes/e011/retarget'),'--out',str(R/'episodes/e011/sim-v1'),'--width','1280','--height','800','--headless','--enable_cameras','--device','cpu','--rendering_mode','performance']
with (R/'episodes/e011/sim-v1.log').open('w') as f:
    result=subprocess.run(cmd,cwd=LAB,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=min(1800,max(1,deadline-time.time())))
print('PILOT_END',result.returncode,flush=True)
