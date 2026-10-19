"""240-second, 2-GiB guard for the new eight-frame CPU source fit."""
from pathlib import Path
import datetime,hashlib,json,os,shutil,subprocess,sys,time
S=Path(__file__).resolve().parents[2];O=S/'e037-RGB-multiview-source-fit-v1';O.mkdir()
worker=Path(__file__).with_name('fit_RGB_multiview_source_v1.py')
report=dict(schema='v2d.cpu-source-probe-guard.v1',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),worker_sha256=hashlib.sha256(worker.read_bytes()).hexdigest(),wall_limit_seconds=240,resident_limit_bytes=2*1024**3,minimum_free_disk_bytes=int(1.5*1024**3),maximum_observed_resident_bytes=0,GPU_work=False)
save=lambda:(O/'guard.json').write_text(json.dumps(report,indent=2)+'\n');start=time.monotonic();save()
with (O/'worker.log').open('w') as log:
    p=subprocess.Popen([sys.executable,'-u',str(worker)],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
    report['worker_pid']=p.pid;save()
    try:
        while p.poll() is None:
            rss=int(subprocess.check_output(['ps','-p',str(p.pid),'-o','rss='],text=True).strip() or 0)*1024
            report['maximum_observed_resident_bytes']=max(report['maximum_observed_resident_bytes'],rss)
            if time.monotonic()-start>240:raise TimeoutError('CPU wall cap')
            if rss>2*1024**3:raise MemoryError('CPU RSS cap')
            if shutil.disk_usage(S).free<int(1.5*1024**3):raise OSError('Disk reserve')
            save();time.sleep(1)
        report['worker_exit_code']=p.returncode
    except BaseException as e:
        report['error']=repr(e);p.terminate()
        try:p.wait(timeout=10)
        except subprocess.TimeoutExpired:p.kill();p.wait(timeout=10)
        report['worker_exit_code']=p.returncode
    finally:
        report.update(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_seconds=time.monotonic()-start);save();print(json.dumps(report),flush=True)
