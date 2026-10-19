"""One serial CPU-only job, strict resources, phase/ledger reconciliation."""
from pathlib import Path
import datetime,hashlib,json,os,shutil,subprocess,sys,time
S=Path(__file__).resolve().parents[1];plan_path=Path(sys.argv[1]);plan=json.loads(plan_path.read_text());O=Path(plan['output']);worker=Path(plan['worker'])
assert not O.exists();assert hashlib.sha256(worker.read_bytes()).hexdigest()==plan['worker_sha256']
for name,digest in plan['dependencies_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
for n in ['phase_state_20261018.json','compute/controller-lifecycle-runtime-ledger-v1.json']:
 p=S/n;j=json.loads(p.read_text());assert not j.get('active_cpu_only_job') and not j.get('active_cpu_work');j['active_cpu_only_job']=str(plan_path);p.write_text(json.dumps(j,indent=2)+'\n')
O.mkdir(mode=0o700);report=dict(schema='v2d.cpu-source-probe-guard.v1',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),worker_sha256=plan['worker_sha256'],plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),wall_limit_seconds=240,resident_limit_bytes=2*1024**3,minimum_free_disk_bytes=int(1.5*1024**3),maximum_observed_resident_bytes=0,GPU_work=False)
save=lambda:(O/'guard.json').write_text(json.dumps(report,indent=2)+'\n');start=time.monotonic();save()
with (O/'worker.log').open('w') as log:
 p=subprocess.Popen([sys.executable,'-u',str(worker)],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'));report['worker_pid']=p.pid;save()
 try:
  while p.poll() is None:
   z=subprocess.run(['ps','-p',str(p.pid),'-o','rss='],capture_output=True,text=True);rss=int(z.stdout.strip() or 0)*1024;report['maximum_observed_resident_bytes']=max(report['maximum_observed_resident_bytes'],rss)
   if time.monotonic()-start>240:raise TimeoutError('CPU wall cap')
   if rss>2*1024**3:raise MemoryError('CPU RSS cap')
   if shutil.disk_usage(S).free<int(1.5*1024**3):raise OSError('Disk reserve')
   save();time.sleep(.5)
  report['worker_exit_code']=p.returncode
 except BaseException as e:
  report['error']=repr(e);p.terminate()
  try:p.wait(timeout=10)
  except subprocess.TimeoutExpired:p.kill();p.wait(timeout=10)
  report['worker_exit_code']=p.returncode
 finally:
  report.update(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_seconds=time.monotonic()-start);save()
  for n in ['phase_state_20261018.json','compute/controller-lifecycle-runtime-ledger-v1.json']:
   pth=S/n;j=json.loads(pth.read_text());assert j['active_cpu_only_job']==str(plan_path);j['active_cpu_only_job']=None;pth.write_text(json.dumps(j,indent=2)+'\n')
  print(json.dumps(report),flush=True)
