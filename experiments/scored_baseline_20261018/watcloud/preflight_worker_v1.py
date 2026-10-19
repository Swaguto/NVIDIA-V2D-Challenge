"""Small read-only runtime probe, submitted through Slurm, never on login."""
import datetime
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess


def emit(stage, value):
    print(json.dumps(dict(stage=stage, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **value)), flush=True)


def run(args, timeout=30, env=None):
    try:
        p = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=timeout, env=env)
        return dict(argv=args, returncode=p.returncode, output=p.stdout[-9000:])
    except Exception as e:
        return dict(argv=args, error=repr(e))


assert os.environ.get('SLURM_JOB_ID'), 'Must run inside a scheduled allocation'
emit('allocation', dict(host=socket.gethostname(), job_id=os.environ['SLURM_JOB_ID'],
    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
    python=run(['python3', '--version']), tmp_free_bytes=shutil.disk_usage('/tmp').free))
emit('gpu', run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,driver_version', '--format=csv,noheader']))
env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
code = """import json,importlib.util
names=['torch','sam2','hamer','numpy','scipy','PIL','newton','warp','jax']
print(json.dumps({name: bool(importlib.util.find_spec(name)) for name in names}))
import torch
print(json.dumps({'torch':torch.__version__,'torch_cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available()}))
if torch.cuda.is_available():
 x=torch.ones(1,device='cuda');torch.cuda.synchronize()
 print(json.dumps({'CUDA_tensor_value':x.item(),'CUDA_device':torch.cuda.get_device_name(0)}))
"""
emit('host_python', run(['python3', '-c', code], timeout=60, env=env))
env.update(XDG_RUNTIME_DIR='/tmp/run', DOCKER_HOST='unix:///tmp/run/docker.sock')
emit('docker_setup', run(['/opt/slurm/bin/slurm-start-dockerd.sh'], timeout=45, env=env))
emit('docker_info', run(['docker', 'info', '--format', '{{json .}}'], env=env))
emit('docker_images', run(['docker', 'image', 'ls', '--format', '{{.Repository}}@{{.Digest}} {{.Size}}'], env=env))
emit('complete', dict(physics_stepped=False, model_inference_executed=False,
    training_steps=0, job_exits_and_releases_resources=True))
