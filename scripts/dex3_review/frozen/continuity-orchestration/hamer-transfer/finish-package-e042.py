import pathlib,json,subprocess,time,hashlib,tarfile
r=pathlib.Path('/workspace/hand-continuity-repair');o=r/'roi-recovery/e042/refit-v1';log=r/'hamer-transfer/refit-e042.log'
while 'REFIT_DONE' not in log.read_text()[-1000:]:
 if time.time()>1792044000:raise RuntimeError('deadline')
 time.sleep(10)
frames=json.load(open(o/'runs/e042/manifest.json'))['frames'];fps=(len(frames)-1)/(frames[-1]['timestamp_s']-frames[0]['timestamp_s'])
v=o/'videos/e042';old=v/'hand-tracking.mp4'
subprocess.run(['/opt/conda/bin/python',str(r/'code/reconstruction/modules/v2d_hamer/lib/render_hands_aligned_video.py'),'--frames_dir',str(o/'runs/e042/align/frames'),'--aligned_dir',str(o/'runs/e042/visual-records'),'--mano_assets_root',str(r/'inputs/issue6-quality-cache/episodes/e042/local-assets/mano'),'--output_path',str(v/'hand-tracking-raw.mp4'),'--fps',str(fps)],check=True)
subprocess.run(['/opt/conda/bin/python',str(r/'code/reconstruction/validation/issue6/label_hand_video.py'),str(v/'hand-tracking-raw.mp4'),str(old)],check=True)
# Never promote a new image estimate into a missing canonical observation.
gate=json.load(open(r/'inputs/issue6-quality-cache/episodes/e042/temporal-hands/e042/observation_gate.json'))['direct_observation'];checked=0
for p in (o/'runs/e042/align/records').rglob('*.json'):
 d=json.load(open(p));side='right' if d['is_right'] else 'left';fid=d['frame_idx']
 if not gate[side][fid]:assert not d['diagnostics']['metric_valid'];checked+=1
folder=o/'review/e042';folder.mkdir(parents=True,exist_ok=False)
files={'cad-temporal-hands.mp4':v/'cad-temporal-hands.mp4','hand-tracking.mp4':old,'baseline-cad-temporal-hands.mp4':r/'all20-results/e042/baseline-videos/e042/cad-temporal-hands.mp4','continuity-receipt.json':o/'runs/e042/continuity-receipt.json','image-recovery-receipt.json':r/'roi-recovery/e042/image-estimates-v2/receipt.json'}
hashes={}
for name,p in files.items():
 dest=folder/name;dest.hardlink_to(p);hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
(folder/'review-hashes.json').write_text(json.dumps(hashes,indent=2));(folder/'authority-check.json').write_text(json.dumps({'missing_observation_records_checked':checked,'new_metric_promotions':0}))
with tarfile.open(r/'all20-transfer/e042-recovered-review.tar','w') as tar:tar.add(folder,arcname='e042')
print('RECOVERED_REVIEW_READY',fps,checked,flush=True)
