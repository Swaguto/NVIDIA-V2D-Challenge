"""Independent local wall/RSS/disk guard; never controls cloud compute."""
import datetime
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

S=Path(__file__).resolve().parents[1]
OUT=S/'fast-stereo-cpu-full-e037-v1';OUT.mkdir()
report=dict(schema='v2d.cpu-model-probe-guard.v1',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),worker_sha256=hashlib.sha256((S/'compute/fast_stereo_cpu_full_e037_v1.py').read_bytes()).hexdigest(),wall_limit_seconds=2700,resident_limit_bytes=3*1024**3,minimum_free_disk_bytes=int(1.5*1024**3),gpu_work=False,maximum_observed_resident_bytes=0)
save=lambda:(OUT/'guard.json').write_text(json.dumps(report,indent=2)+'\n')
started=time.monotonic();save()
with (OUT/'worker.log').open('w') as log:
    child=subprocess.Popen([sys.executable,'-u',str(S/'compute/fast_stereo_cpu_full_e037_v1.py')],stdout=log,stderr=subprocess.STDOUT,cwd=S)
    report['worker_pid']=child.pid;save()
    try:
        while child.poll() is None:
            elapsed=time.monotonic()-started
            rss=subprocess.run(['ps','-p',str(child.pid),'-o','rss='],capture_output=True,text=True,timeout=5)
            resident=int(rss.stdout.strip() or 0)*1024
            report['maximum_observed_resident_bytes']=max(report['maximum_observed_resident_bytes'],resident)
            if elapsed>report['wall_limit_seconds']:raise TimeoutError('CPU probe wall cap')
            if resident>report['resident_limit_bytes']:raise MemoryError('CPU probe RSS cap')
            if shutil.disk_usage(S).free<report['minimum_free_disk_bytes']:raise OSError('CPU probe disk reserve')
            save();time.sleep(1)
        report['worker_exit_code']=child.returncode
        if child.returncode:report['error']='Worker failed; retained log and partial reports'
    except BaseException as error:
        report['error']=repr(error);child.terminate()
        try:child.wait(timeout=10)
        except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)
        report['worker_exit_code']=child.returncode
    finally:
        report['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();report['wall_seconds']=time.monotonic()-started;save()
        print(json.dumps(report),flush=True)
