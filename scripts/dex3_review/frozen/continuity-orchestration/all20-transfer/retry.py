import pathlib,time,subprocess,tarfile,json,datetime,os,hashlib
root=pathlib.Path('/workspace/hand-continuity-repair'); transfer=root/'all20-transfer';status={};deadline=datetime.datetime.fromisoformat('2026-10-15T06:00:00+00:00').timestamp()
env=os.environ.copy();env.update(INPUT_ROOT=str(root/'inputs'),PYTHON='/opt/conda/bin/python',PYTHONPATH=str(root/'runtime-additions'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
episodes=[0,2]
while len(status)<len(episodes) and time.time()<deadline:
 progress=False
 for ep in episodes:
  label=f'e{ep:03d}'
  if label in status:continue
  archive=transfer/(label+'.tar.gz')
  if not archive.exists() or time.time()-archive.stat().st_mtime<8:continue
  if subprocess.run(['gzip','-t',str(archive)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:continue
  h=hashlib.sha256()
  with archive.open('rb') as f:
   while chunk:=f.read(8*1024**2):h.update(chunk)
  print('EXTRACT',label,h.hexdigest(),flush=True)
  with tarfile.open(archive) as t:
   for item in t:
    if item.name.startswith('/') or '..' in pathlib.PurePosixPath(item.name).parts:raise ValueError('unsafe archive')
    t.extract(item,root/'inputs')
  out=root/'all20-retry'/label;env['OUT']=str(out)
  print('RUN',label,flush=True)
  with (transfer/(label+'-retry-run.log')).open('w') as log:
   try:r=subprocess.run(['bash',str(root/'code/reconstruction/validation/issue6/run_hand_continuity_review.sh'),str(ep)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,deadline-time.time()));rc=r.returncode
   except subprocess.TimeoutExpired:rc=124
  status[label]={'exit_code':rc,'archive_sha256':h.hexdigest(),'output':str(out)}
  (transfer/'retry-status.json').write_text(json.dumps(status,indent=2));print('FINISH',label,rc,flush=True);progress=True
 if not progress:time.sleep(10)
print('CONTROLLER_DONE',len(status),flush=True)
