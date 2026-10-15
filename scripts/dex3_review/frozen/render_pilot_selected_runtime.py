import datetime
import json
import os
from pathlib import Path
import subprocess
import time

R=Path('/workspace/hand-continuity-repair/dex3-isaac-review')
deadline=datetime.datetime.fromisoformat('2026-10-15T05:45:00+00:00').timestamp()
options=[(R/'runtime-install.log',R/'runtime6',R/'IsaacLab3'),
         (R/'runtime-local-install.log',Path('/opt/v2d-dex3-review/runtime6'),Path('/opt/v2d-dex3-review/IsaacLab3'))]
while time.time()<deadline:
    selected=None
    for path,prefix,lab in options:
        if path.exists():
            log=path.read_text()
            if log.rstrip().endswith('}') and '\"status\": \"passed\"' in log[-900:]:
                selected=(path,prefix,lab)
                break
    if selected and (R/'episodes/e011/retarget/quality.json').is_file():break
    time.sleep(5)
else:raise RuntimeError('Runtime preparation deadline')
_,prefix,LAB=selected
PY=prefix/'bin/python'
# Stop only the other software installer launched by this review.
import signal
other=next(p for _,p,_ in options if p!=prefix)
for p in Path('/proc').iterdir():
    if p.name.isdigit():
        try:
            argv=(p/'cmdline').read_bytes().split(b'\0')
            if (argv[0]==str(other/'bin/python').encode() and b'pip' in argv) or (b'--prefix' in argv and argv[argv.index(b'--prefix')+1]==str(other).encode()):
                os.kill(int(p.name),signal.SIGTERM)
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
env=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',OMNI_KIT_ALLOW_ROOT='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
env.pop('PYTHONPATH',None)
print('SIM_RUNTIME_READY',flush=True)
with (R/'runtime-ready.json').open('w') as f:json.dump({'lock':str(prefix/'replay-requirements.txt'),'python':str(PY),'lab':str(LAB),'installer_check':'passed'},f)
cmd=[str(PY),str(R/'replay_continuity_dex3_sim6.py'),'--source',str(R/'retarget-code/robotic_grounding'),'--bundle',str(R/'episodes/e011/bundle'),'--retarget',str(R/'episodes/e011/retarget'),'--out',str(R/'episodes/e011/sim-v1'),'--width','1280','--height','800','--headless','--enable_cameras','--device','cpu','--rendering_mode','performance']
with (R/'episodes/e011/sim-v1.log').open('w') as f:
    result=subprocess.run(cmd,cwd=LAB,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=min(1800,max(1,deadline-time.time())))
print('PILOT_END',result.returncode,flush=True)
