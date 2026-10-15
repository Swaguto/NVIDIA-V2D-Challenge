import pathlib,json,subprocess,time,hashlib,tarfile
r=pathlib.Path('/workspace/hand-continuity-repair');o=r/'roi-recovery/e041/refit-v1';log=r/'hamer-transfer/refit-e041.log'
while 'REFIT_DONE' not in log.read_text()[-1000:]:
 if time.time()>1792044000:raise RuntimeError('deadline')
 time.sleep(10)
frames=json.load(open(o/'runs/e041/manifest.json'))['frames'];fps=(len(frames)-1)/(frames[-1]['timestamp_s']-frames[0]['timestamp_s'])
v=o/'videos/e041';raw=v/'hand-tracking-raw.mp4';correct=v/'hand-tracking-source-rate.mp4'
subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(raw),'-vf',f'setpts={30/fps}*PTS','-r',str(fps),'-vsync','0','-c:v','libx264','-crf','18',str(correct)],check=True)
old=v/'hand-tracking.mp4';old.rename(v/'hand-tracking-wrong-rate.mp4')
subprocess.run(['/opt/conda/bin/python',str(r/'code/reconstruction/validation/issue6/label_hand_video.py'),str(correct),str(old)],check=True)
# Never promote a new image estimate into a missing canonical observation.
gate=json.load(open(r/'inputs/issue6-quality-cache/episodes/e041/temporal-hands/e041/observation_gate.json'))['direct_observation'];checked=0
for p in (o/'runs/e041/align/records').rglob('*.json'):
 d=json.load(open(p));side='right' if d['is_right'] else 'left';fid=d['frame_idx']
 if not gate[side][fid]:assert not d['metric_valid'];checked+=1
folder=o/'review/e041';folder.mkdir(parents=True,exist_ok=False)
files={'cad-temporal-hands.mp4':v/'cad-temporal-hands.mp4','hand-tracking.mp4':old,'baseline-cad-temporal-hands.mp4':r/'all20-results/e041/baseline-videos/e041/cad-temporal-hands.mp4','continuity-receipt.json':o/'runs/e041/continuity-receipt.json','image-recovery-receipt.json':r/'roi-recovery/e041/fresh-observations-v3/receipt.json'}
hashes={}
for name,p in files.items():
 dest=folder/name;dest.hardlink_to(p);hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
(folder/'review-hashes.json').write_text(json.dumps(hashes,indent=2));(folder/'authority-check.json').write_text(json.dumps({'missing_observation_records_checked':checked,'new_metric_promotions':0}))
with tarfile.open(r/'all20-transfer/e041-recovered-review.tar','w') as tar:tar.add(folder,arcname='e041')
print('RECOVERED_REVIEW_READY',fps,checked,flush=True)
