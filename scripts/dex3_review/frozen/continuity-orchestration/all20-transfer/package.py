import pathlib,json,tarfile,subprocess,os,hashlib
root=pathlib.Path('/workspace/hand-continuity-repair');transfer=root/'all20-transfer';statuses={}
for name in ['status.json','retry-status.json','parallel-status.json','worker-c-status.json','worker-d-status.json']:
 p=transfer/name
 if p.exists():statuses.update(json.loads(p.read_text()))
for label,row in statuses.items():
 out=pathlib.Path(row['output']);archive=transfer/(label+'-review.tar');
 if row['exit_code']!=0 and not (out/'runs'/label/'continuity-receipt.json').is_file():continue
 raw=out/'videos'/label/'hand-tracking-raw.mp4';labeled=out/'videos'/label/'hand-tracking.mp4'
 if not labeled.is_file() and raw.is_file():
  subprocess.run(['/opt/conda/bin/python',str(root/'code/reconstruction/validation/issue6/label_hand_video.py'),str(raw),str(labeled)],check=True,stdout=subprocess.DEVNULL)
 if not labeled.is_file():continue
 baseline=out/'baseline-videos'/label/'cad-temporal-hands.mp4'
 if archive.exists() and (baseline.parent/'cad-temporal-hands-receipt.json').is_file() and archive.stat().st_mtime>=baseline.stat().st_mtime:continue
 if not (baseline.parent/'cad-temporal-hands-receipt.json').is_file():
  env=os.environ.copy();env.update(PYTHONPATH=str(root/'code/reconstruction/modules/v2d_issue6')+':'+str(root/'runtime-additions'),PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
  cmd=['/opt/conda/bin/python',str(root/'code/reconstruction/validation/issue6/render_combined_overlay.py'),'--episode',str(int(label[1:])),'--run-root',str(root/'inputs/issue6-quality-cache/episodes'/label/'issue6-full-runs/all20-final-v3'),'--review-root',str(out/'baseline-videos'),'--manifest',str(out/'runs'/label/'manifest.json'),'--modules-dir',str(root/'code/reconstruction/modules'),'--mano-assets-root',str(root/'inputs/issue6-quality-cache/episodes'/label/'local-assets/mano'),'--hands-mode','metric']
  with (transfer/(label+'-baseline.log')).open('w') as log:r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
  if r.returncode:print('BASELINE_FAIL',label,flush=True);continue
 files={'continuity-receipt.json':out/'runs'/label/'continuity-receipt.json','cad-temporal-hands.mp4':out/'videos'/label/'cad-temporal-hands.mp4','hand-tracking.mp4':out/'videos'/label/'hand-tracking.mp4','baseline-cad-temporal-hands.mp4':baseline}
 hashes={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in files.items()}
 receipt=transfer/(label+'-review-hashes.json');receipt.write_text(json.dumps(hashes,indent=2));files['review-hashes.json']=receipt
 with tarfile.open(str(archive)+'.partial','w') as t:
  for n,p in files.items():t.add(p,arcname=label+'/'+n,recursive=False)
 pathlib.Path(str(archive)+'.partial').rename(archive);print('PACKAGED',label,archive.stat().st_size,flush=True)
