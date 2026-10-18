"""Independent local resource guard for one full native object/hand diagnostic."""
import datetime
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

S = Path(__file__).resolve().parents[1]
OUT = S / 'plane-aligned-grasp-separation-v2'
OUT.mkdir()
worker = S / 'compute/audit_aligned_grasp_separation_v2.py'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
report = dict(schema='v2d.cpu-model-probe-guard.v1',
    started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    producer_sha256=sha(Path(__file__)), worker_sha256=sha(worker),
    wall_limit_seconds=300, resident_limit_bytes=2*1024**3,
    minimum_free_disk_bytes=int(1.5*1024**3), gpu_work=False,
    maximum_observed_resident_bytes=0)
save = lambda: (OUT / 'guard.json').write_text(json.dumps(report, indent=2)+'\n')
started = time.monotonic()
save()
with (OUT / 'worker.log').open('w') as log:
    child = subprocess.Popen([sys.executable, '-u', str(worker)], stdout=log,
                             stderr=subprocess.STDOUT, cwd=S)
    report['worker_pid'] = child.pid
    save()
    try:
        while child.poll() is None:
            rss = subprocess.run(['ps', '-p', str(child.pid), '-o', 'rss='],
                                 capture_output=True, text=True, timeout=5)
            resident = int(rss.stdout.strip() or 0)*1024
            report['maximum_observed_resident_bytes'] = max(report['maximum_observed_resident_bytes'], resident)
            if time.monotonic()-started > report['wall_limit_seconds']:
                raise TimeoutError('CPU native audit wall cap')
            if resident > report['resident_limit_bytes']:
                raise MemoryError('CPU native audit RSS cap')
            if shutil.disk_usage(S).free < report['minimum_free_disk_bytes']:
                raise OSError('CPU native audit disk reserve')
            save()
            time.sleep(1)
        report['worker_exit_code'] = child.returncode
        if child.returncode:
            report['error'] = 'Worker failed; partial files retained'
    except BaseException as error:
        report['error'] = repr(error)
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=10)
        report['worker_exit_code'] = child.returncode
    finally:
        report['completed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        report['wall_seconds'] = time.monotonic()-started
        save()
        print(json.dumps(report), flush=True)
