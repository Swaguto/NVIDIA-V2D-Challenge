"""Fresh isolated dependencies, private inputs and bounded inference outputs."""
import hashlib
import json
import subprocess
import tarfile
import time
from pathlib import Path

B = Path('/data/v2d-scored')
ROOT = B/'learned-stereo-probe-v3'
OUT = ROOT/'outputs'
OUT.mkdir()
D = ['docker','--host','unix:///data/v2d-scored/docker.sock']
IMAGE = 'ghcr.io/swaguto/nvidia-v2d-challenge-dex3-review@sha256:6e2039d0a54e14a188a1798efe57c3465d002f063c782c827b791e4418d2dce6'
container = 'v2d-learned-stereo-probe-v3'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
receipt = dict(schema='v2d.private-stereo-gpu-probe.v1', image=IMAGE,
               new_training_steps=0, source_inputs_modified=False,
               private_assets_published=False, physical_task_accepted=False)
save = lambda: (OUT/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
save()
try:
    if subprocess.run(D+['info'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        with (B/'dockerd.log').open('ab') as log:
            subprocess.Popen(['sudo','-n','dockerd','--host',D[2],'--data-root',str(B/'docker'),
                              '--exec-root','/run/v2d-scored-docker','--pidfile',str(B/'docker.pid'),
                              '--bridge','none','--iptables=false','--ip-forward=false','--ip-masq=false',
                              '--add-runtime','nvidia=/usr/bin/nvidia-container-runtime'],
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(30):
            if not subprocess.run(D+['info'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
                break
            time.sleep(1)
        else:
            raise TimeoutError('Perception Docker daemon unavailable')
    running = subprocess.check_output(D+['ps','--format','{{.Names}}'], text=True).strip()
    assert not running, 'Other perception container is active: '+running
    # Dependency selection matches cuDNN major; install into this job only.
    bootstrap = '''import json,subprocess,sys,torch
v = torch.backends.cudnn.version()
runtime = '1.20.1' if v >= 90000 else '1.18.0'
print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,cudnn=v,ort_selected=runtime)),flush=True)
subprocess.run([sys.executable,'-m','pip','install','--no-cache-dir','--no-deps','--target','/job/deps','onnxruntime-gpu=='+runtime],check=True,timeout=240)
subprocess.run([sys.executable,'-m','pip','freeze'],stdout=open('/job/outputs/base-runtime.freeze.txt','w'),check=True)
subprocess.run([sys.executable,'-u','/job/inference.py'],check=True,timeout=900)
'''
    with (OUT/'inference.log').open('wb') as log:
        result = subprocess.run(D+['run','--rm','--name',container,'--gpus','all','--network','host',
                                  '--ipc','host','-v',str(ROOT)+':/job','--entrypoint','/opt/conda/bin/python',
                                  IMAGE,'-u','-c',bootstrap], stdout=log, stderr=subprocess.STDOUT, timeout=1200)
        receipt['inference_returncode'] = result.returncode
    assert result.returncode == 0, 'Depth inference failed; see retained log'
except BaseException as error:
    receipt['error'] = repr(error)
    raise
finally:
    subprocess.run(D+['rm','-f',container], capture_output=True, timeout=30)
    save()
    inventory = {str(p.relative_to(OUT)):dict(size=p.stat().st_size,sha256=sha(p)) for p in OUT.rglob('*') if p.is_file() and not p.is_symlink()}
    (ROOT/'inventory.json').write_text(json.dumps(inventory, indent=2)+'\n')
    with tarfile.open(ROOT/'results.tar.gz','w:gz',compresslevel=1) as archive:
        for name in inventory:
            archive.add(OUT/name, arcname=name, recursive=False)
