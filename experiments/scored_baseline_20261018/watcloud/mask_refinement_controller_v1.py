"""Local owner controller for a bounded Watcloud source-only inference probe."""
import base64
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import threading
import time

S=Path(__file__).resolve().parents[2]
W=Path(__file__).resolve().parent
R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
OUT=S/'e037-watcloud-mask-refinement-v1'
ROOT='/tmp/v2d-fresh-hand'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','watcloud']
job_id=None
finished=threading.Event()

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(p,data):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(p)

def update(stage,**data):
    state=dict(stage=stage,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),job_id=job_id,**data)
    save(W/'mask-refinement-controller-state-v1.json',state)
    phase_path=S/'phase_state_20261018.json';phase=json.loads(phase_path.read_text())
    phase['updated_utc']=state['utc'];phase['preferred_compute']='watcloud';phase['watcloud_current']=state
    phase['active_gpu']=None if stage in ('finished','failed') else dict(provider='watcloud',job_id=job_id,stage=stage)
    save(phase_path,phase)
    ledger_path=S/'compute/controller-lifecycle-runtime-ledger-v1.json';ledger=json.loads(ledger_path.read_text())
    ledger['latest_update_utc']=state['utc'];ledger['preferred_compute']='watcloud';ledger['watcloud_current']=state
    save(ledger_path,ledger)
    print(json.dumps(state),flush=True)

def step(code,stdin=None,stdout=subprocess.PIPE,timeout=900):
    command='export PATH=/opt/slurm/bin:/opt/slurm/sbin:$PATH; srun --jobid='+job_id+' --overlap --ntasks=1 --cpus-per-task=1 python3 -c '+shlex.quote(code)
    return subprocess.run(SSH+[command],input=stdin,stdout=stdout,stderr=subprocess.PIPE,timeout=timeout)

def guard():
    if not finished.wait(1100) and job_id:
        subprocess.run(SSH+['/opt/slurm/bin/scancel '+job_id],capture_output=True,timeout=30)
        save(W/'mask-refinement-controller-timeout-v1.json',dict(job_id=job_id,deadline_s=1100,cancel_sent=True))

assert not OUT.exists();OUT.mkdir(mode=0o700)
rows=[]
sam=s_path=S/'models/sam2.1_hiera_large.pt'
rows.append((sam,'weights/sam2.1_hiera_large.pt','2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318'))
base=S/'e037-fresh-hand-probe-payload-v1/source-inputs.private.tar.gz'
assert sha(base)=='04d0b14fec9ad2e9665690a7589dae2c32f3e10d35dbc8eeb44172fa8e3c941c'
rows.append((base,base.name,'04d0b14fec9ad2e9665690a7589dae2c32f3e10d35dbc8eeb44172fa8e3c941c'))
payload=S/'e037-watcloud-mask-refinement-payload-v1'
for name,expected in [('committed-refinement-code-v1.tar.gz','cce6bcf3b3b044ddf398aed4b441218d4d6d229792a5c9a011dc511a5ea7f166'),('refinement-inputs.private.tar.gz','b9e6cdf9e467560be33dbdf19e4a23f1e58363f96f8d6c8b90068684e4460f27')]:
    assert sha(payload/name)==expected
    rows.append((payload/name,name,expected))
manifest={'files':{dst:dict(sha256=expected,bytes=src.stat().st_size) for src,dst,expected in rows}}
save(W/'mask-refinement-transfer-manifest-v1.json',manifest)
token=subprocess.check_output(['gh','auth','token'],text=True).strip()
worker=base64.b64encode((W/'mask_refinement_worker_v1.py').read_bytes()).decode()
remote='export PATH=/opt/slurm/bin:/opt/slurm/sbin:$PATH; exec srun --immediate=60 --job-name=v2d-e037-mask-only --time=00:20:00 --ntasks=1 --cpus-per-task=2 --mem=12G --gres=shard:rtx_3090:8192,tmpdisk:61440 --unbuffered python3 -u -c '+shlex.quote('import base64;exec(compile(base64.b64decode('+repr(worker)+'),"watcloud-fresh-hand-worker","exec"))')
proc=subprocess.Popen(SSH+[remote],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
proc.stdin.write(json.dumps(dict(username='Swaguto',token=token))+'\n');proc.stdin.close();token=None
threading.Thread(target=guard,daemon=True).start()
update('allocating',controller_pid=os.getpid(),allocation_limit_s=1200,controller_soft_limit_s=1100)

def transfer():
    try:
        code='import tarfile; t=tarfile.open(fileobj=__import__("sys").stdin.buffer,mode="r|"); t.extractall('+repr(ROOT)+'); print("private-input-transfer-complete")'
        command='export PATH=/opt/slurm/bin:/opt/slurm/sbin:$PATH; srun --jobid='+job_id+' --overlap --ntasks=1 --cpus-per-task=1 python3 -c '+shlex.quote(code)
        with (W/'mask-refinement-input-transfer-v1.log').open('wb') as log:
            p=subprocess.Popen(SSH+[command],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT)
            with tarfile.open(fileobj=p.stdin,mode='w|') as t:
                for src,dst,expected in rows:t.add(src,arcname=dst,recursive=False)
                content=(json.dumps(manifest)+'\n').encode();info=tarfile.TarInfo('transfer-complete.json');info.size=len(content);info.mode=0o600;t.addfile(info,io.BytesIO(content))
            p.stdin.close()
            if p.wait(timeout=60):raise RuntimeError('Private input transfer failed')
        save(W/'mask-refinement-input-transfer-v1.json',dict(job_id=job_id,complete=True,bytes=sum(p['bytes'] for p in manifest['files'].values()),verified_by_worker_before_use=True))
    except BaseException as e:save(W/'mask-refinement-input-transfer-error-v1.json',dict(error=repr(e),job_id=job_id))

try:
    with (W/'mask-refinement-session-v1.jsonl').open('w') as log:
        for line in proc.stdout:
            log.write(line);log.flush()
            try:event=json.loads(line)
            except ValueError:continue
            stage=event.get('stage')
            if stage=='allocation':job_id=str(event['job_id']);update('allocated',host=event['host'],allocation_limit_s=1200,controller_soft_limit_s=1100)
            elif stage=='awaiting_input_transfer':
                update('staging_private_inputs_and_pulling_pinned_image');threading.Thread(target=transfer,daemon=True).start()
            elif stage in ('container_CUDA_passed','inputs_verified','inference_started'):update(stage)
            elif stage=='awaiting_backup':
                update('backing_up_private_outputs',worker_complete=event['complete'])
                with (OUT/'outputs.private.tar.gz').open('wb') as archive:
                    p=step('import tarfile,sys;t=tarfile.open(fileobj=sys.stdout.buffer,mode="w|gz");t.add('+repr(ROOT+'/outputs')+',arcname="outputs");t.close()',stdout=archive,timeout=180)
                if p.returncode:raise RuntimeError('Private backup transfer failed: '+p.stderr.decode()[-1000:])
                with tarfile.open(OUT/'outputs.private.tar.gz') as t:
                    for m in t:
                        if Path(m.name).is_absolute() or '..' in Path(m.name).parts or not (m.isfile() or m.isdir()):raise ValueError('Unsafe output archive')
                        t.extract(m,OUT)
                receipt=json.loads((OUT/'outputs/session-receipt.json').read_text())
                for name,entry in receipt['files'].items():
                    f=OUT/'outputs'/name;assert f.stat().st_size==entry['bytes'] and sha(f)==entry['sha256'],name
                backup=dict(job_id=job_id,files_verified=len(receipt['files']),archive_sha256=sha(OUT/'outputs.private.tar.gz'),worker_complete=receipt['complete'])
                save(W/'mask-refinement-private-backup-verified-v1.json',backup)
                ack=step('from pathlib import Path;Path('+repr(ROOT+'/backup-confirmed.json')+').write_text('+repr(json.dumps(backup))+')',timeout=30)
                assert ack.returncode==0,ack.stderr
        rc=proc.wait(timeout=30)
    if not job_id:raise RuntimeError('Slurm allocation did not start; inspect session log')
    status=subprocess.run(SSH+['/opt/slurm/bin/sacct -j '+job_id+' --noheader --parsable2 --format=JobID,State,ExitCode,Elapsed,AllocTRES'],capture_output=True,text=True,timeout=30)
    save(W/'mask-refinement-scheduler-finish-v1.json',dict(job_id=job_id,controller_returncode=rc,accounting=status.stdout,accounting_stderr=status.stderr))
    update('finished' if rc==0 else 'failed',controller_returncode=rc,private_output=str(OUT),scheduler_receipt=str(W/'mask-refinement-scheduler-finish-v1.json'))
except BaseException as e:
    if job_id:subprocess.run(SSH+['/opt/slurm/bin/scancel '+job_id],capture_output=True,timeout=30)
    update('failed',error=repr(e))
    raise
finally:
    finished.set()
