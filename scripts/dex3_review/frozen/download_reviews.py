import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile
import time

GALLERY=Path('/Users/rohithiruku/v2d_episode_previews/recovered_20261014/continuity_fix/dex3_isaac')
CACHE=Path('/Users/rohithiruku/dex3_isaac_episode_review_20261014/downloads');CACHE.mkdir(exist_ok=True)
EPISODES=[2,11,12,13,18,21,23,24,31,40,41,42]
SSH=['ssh','-i','/Users/rohithiruku/.ssh/id_ed25519','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','ConnectTimeout=15','-p','22086','root@69.30.85.27']
SCP=['scp','-q','-i','/Users/rohithiruku/.ssh/id_ed25519','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','ConnectTimeout=15','-P','22086']
REMOTE='/workspace/hand-continuity-repair/dex3-isaac-review'
remote_script="""/opt/conda/bin/python - <<'PYREMOTE'
import json,pathlib
r=pathlib.Path('/workspace/hand-continuity-repair/dex3-isaac-review')
print(json.dumps([p.name for p in r.glob('e*-review.tar') if p.stat().st_size>0]))
PYREMOTE"""

def results():
    rows=[]
    for ep in EPISODES:
        label=f'e{ep:03d}';p=GALLERY/label
        row=dict(episode=label,ready=False,objects=[])
        if (p/'delivery-hashes.json').exists():
            receipt=json.load(open(p/'isaac-receipt.json'));validation=json.load(open(p/'bundle-validation.json'));quality=json.load(open(p/'retarget-quality.json'));media=json.load(open(p/'media-verification.json'))
            row.update(ready=True,frames=receipt['source_frames'],duration=float(media['comparison.mp4']['duration']),objects=validation['objects'],visual_counts=receipt['rendered_inferred_robot_state_counts'],metric_counts=validation['source_metric_hand_frames'],tip_p95_mm=[quality['per_hand'][s]['three_tip_position_error_mm']['p95'] for s in ['left','right']],review_note='Full clips rendered and source frame counts verified; visual review pending.')
            if (p/'visual-review.json').exists():row['review_note']=json.load(open(p/'visual-review.json'))['summary']
        rows.append(row)
    temp=GALLERY/'results.json.tmp';temp.write_text(json.dumps(rows,indent=2));temp.replace(GALLERY/'results.json')
    return [row['episode'] for row in rows if row['ready']]


deadline=datetime.datetime.fromisoformat('2026-10-15T06:00:00+00:00').timestamp()
backoff=10
while time.time()<deadline:
    complete=results()
    if len(complete)==len(EPISODES):break
    try:
        ready=json.loads(subprocess.check_output(SSH+[remote_script],text=True,timeout=30))
        progress=False
        for name in ready:
            if not re.fullmatch(r'e\d{3}-review\.tar',name):raise ValueError('Unexpected archive name')
            label=name[:4]
            if label in complete:continue
            temp=CACHE/(name+'.partial')
            subprocess.run(SCP+['root@69.30.85.27:'+REMOTE+'/'+name,str(temp)],check=True,timeout=180)
            staging=CACHE/(label+'-staging')
            if staging.exists():staging.rename(CACHE/(label+'-staging-old-'+str(time.time_ns())))
            staging.mkdir(exist_ok=False)
            with tarfile.open(temp) as tar:
                for member in tar.getmembers():
                    path=Path(member.name)
                    if path.is_absolute() or '..' in path.parts or not (member.isdir() or member.isfile()) or path.parts[0]!=label:raise ValueError('Unsafe review archive')
                tar.extractall(staging)
            folder=staging/label;hashes=json.load(open(folder/'delivery-hashes.json'))
            verification=json.load(open(folder/'media-verification.json'))
            if verification['comparison.mp4'].get('presentation_timestamps_strictly_increasing') is not True:raise ValueError('Waiting for corrected presentation timestamps')
            for file,digest in hashes.items():
                if Path(file).name!=file or hashlib.sha256((folder/file).read_bytes()).hexdigest()!=digest:raise ValueError('Review transfer hash mismatch')
            (folder/'transport-verification.json').write_text(json.dumps({'all_payload_sha256_matches_primary':True,'verified_files':len(hashes)},indent=2))
            folder.rename(GALLERY/label);temp.rename(CACHE/name);staging.rmdir();progress=True
            print('DOWNLOAD_READY',label,'verified',len(hashes),'files',flush=True);results()
        backoff=10 if progress else min(60,backoff+10)
    except Exception as error:
        print('DOWNLOAD_RETRY',type(error).__name__,str(error)[:250],flush=True);backoff=min(60,backoff+10)
    time.sleep(backoff)
print('DOWNLOAD_BATCH_END',results(),flush=True)
