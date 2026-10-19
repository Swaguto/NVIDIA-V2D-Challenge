"""One Slurm scratch session; private inputs and registry auth never enter images."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tarfile
import time

IMAGE='ghcr.io/swaguto/nvidia-v2d-challenge-dex3-review@sha256:6e2039d0a54e14a188a1798efe57c3465d002f063c782c827b791e4418d2dce6'
ROOT=Path('/tmp/v2d-fresh-hand')
ENV=dict(os.environ,DOCKER_HOST='unix:///tmp/run/docker.sock',XDG_RUNTIME_DIR='/tmp/run',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
NAME='v2d-fresh-hand-e037'

def emit(stage,**data):
    print(json.dumps(dict(stage=stage,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**data)),flush=True)

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def run(args,log,timeout):
    with (ROOT/'outputs'/log).open('w') as out:
        p=subprocess.run(args,env=ENV,stdout=out,stderr=subprocess.STDOUT,timeout=timeout)
    if p.returncode:raise RuntimeError(f'{log} exited {p.returncode}')

def extract(archive,dest):
    dest.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive) as t:
        for member in t:
            name=Path(member.name)
            if name.is_absolute() or '..' in name.parts or not (member.isfile() or member.isdir()):
                raise ValueError('Archive must contain regular relative files only')
            t.extract(member,dest)

assert os.environ.get('SLURM_JOB_ID')
ROOT.mkdir(mode=0o700)
(ROOT/'outputs').mkdir()
emit('allocation',job_id=os.environ['SLURM_JOB_ID'],host=socket.gethostname(),root=str(ROOT),
    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
secret=json.loads(sys.stdin.readline())
receipt=dict(schema='v2d.watcloud-fresh-hand-session.v1',job_id=os.environ['SLURM_JOB_ID'],
    host=socket.gethostname(),image=IMAGE,model_inference_executed=False,complete=False,
    training_steps=0,physics_stepped=False,reference_accepted=False)
auth=ROOT/'registry-auth';auth.mkdir(mode=0o700)
try:
    run(['/opt/slurm/bin/slurm-start-dockerd.sh'],'docker-setup.log',45)
    emit('awaiting_input_transfer',root=str(ROOT))
    login=subprocess.run(['docker','--config',str(auth),'login','ghcr.io','--username',secret['username'],'--password-stdin'],
        input=secret.pop('token'),text=True,env=ENV,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
    secret.clear()
    if login.returncode:raise RuntimeError('Private registry authentication failed')
    run(['docker','--config',str(auth),'pull',IMAGE],'image-pull.log',900)
    shutil.rmtree(auth)
    probe='import torch,json; assert torch.cuda.is_available(); x=torch.ones(1,device="cuda"); torch.cuda.synchronize(); print(json.dumps({"torch":torch.__version__,"device":torch.cuda.get_device_name(0),"capability":torch.cuda.get_device_capability(0),"value":x.item()}))'
    run(['docker','run','--rm','--pull=never','--network=none','--gpus','all','--entrypoint','/workspace/hand-runtime/bin/python',IMAGE,'-c',probe],'container-CUDA-probe.log',120)
    emit('container_CUDA_passed')
    deadline=time.monotonic()+1200
    while not (ROOT/'transfer-complete.json').is_file():
        if time.monotonic()>deadline:raise TimeoutError('Input transfer deadline exceeded')
        time.sleep(2)
    manifest=json.loads((ROOT/'transfer-complete.json').read_text())
    for name,entry in manifest['files'].items():
        p=ROOT/name
        if p.stat().st_size!=entry['bytes'] or sha(p)!=entry['sha256']:raise ValueError('Transferred input hash mismatch: '+name)
    receipt['input_files_verified']=manifest['files']
    extract(ROOT/'source-inputs.private.tar.gz',ROOT/'source')
    extract(ROOT/'committed-perception-code-v2.tar.gz',ROOT/'code-context')
    context=json.loads((ROOT/'code-context/context-manifest.json').read_text())
    assert context['source_commit']=='60c943f0c50af779a11ca7ff79ed8642fd966e59'
    for name,expected in context['files'].items():
        assert sha(ROOT/'code-context'/name)==expected,name
    receipt['source_commit']=context['source_commit']
    emit('inputs_verified',files=len(manifest['files']),source_commit=context['source_commit'])
    command=['docker','run','--rm','--pull=never','--network=none','--gpus','all','--shm-size=8g','--name',NAME,
        '-v',str(ROOT/'source/inputs')+':/Users/rohithiruku/v2d_scored_baseline_20261017:ro',
        '-v',str(ROOT/'code-context/code')+':/workspace/pr32:ro',
        '-v',str(ROOT/'weights/hamer')+':/workspace/hamer-weights:ro',
        '-v',str(ROOT/'weights/sam2.1_hiera_large.pt')+':/workspace/pr32-models/sam2.1_hiera_large.pt:ro',
        '-v',str(ROOT/'outputs')+':/outputs','-e','OMP_NUM_THREADS=4','-e','OPENBLAS_NUM_THREADS=4',
        '--entrypoint','/workspace/hand-runtime/bin/python',IMAGE,
        '/workspace/pr32/scripts/track3/recover_source_hand.py','--request',
        '/Users/rohithiruku/v2d_scored_baseline_20261017/e037-fresh-source-hand-request-v2/request.json',
        '--out','/outputs/recovery','--sam-weights','/workspace/pr32-models/sam2.1_hiera_large.pt',
        '--hamer-weights','/workspace/hamer-weights']
    receipt['inference_argv']=command
    receipt['model_inference_executed']=True
    emit('inference_started',worker_timeout_s=900)
    run(command,'source-hand-inference.log',900)
    receipt['complete']=True
except BaseException as e:
    receipt['error']=repr(e)
    emit('worker_error',error=repr(e))
finally:
    secret.clear()
    if auth.exists():shutil.rmtree(auth)
    subprocess.run(['docker','rm','-f',NAME],env=ENV,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=30)
    receipt['files']={str(p.relative_to(ROOT/'outputs')):dict(sha256=sha(p),bytes=p.stat().st_size)
        for p in sorted((ROOT/'outputs').rglob('*')) if p.is_file()}
    (ROOT/'outputs/session-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    emit('awaiting_backup',complete=receipt['complete'],error=receipt.get('error'))
    end=time.monotonic()+300
    while not (ROOT/'backup-confirmed.json').is_file() and time.monotonic()<end:time.sleep(2)
    emit('allocation_releasing',backup_marker_present=(ROOT/'backup-confirmed.json').is_file())
sys.exit(0 if receipt['complete'] else 1)
