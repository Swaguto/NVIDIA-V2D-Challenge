"""Stage frozen automatic hand observations for an independent refinement run."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main(a):
    kit=a.kit.resolve();episode=a.episode_root.resolve();out=a.out.resolve()
    if out.exists():raise FileExistsError(out)
    inventory=json.loads((kit/'artifact-manifest.json').read_text())
    for name,record in inventory['files'].items():
        if sha(kit/name)!=record['sha256']:raise ValueError('Changed reference file: '+name)
    source=kit/'perception-checkpoint'
    metadata=json.loads((source/'inputs.json').read_text())
    proposal=json.loads((episode/'proposals/receipt.json').read_text())
    quality=json.loads((episode/'quality/summary.json').read_text())
    if proposal['episode']!=metadata['episode'] or quality['episode']!=metadata['episode']:
        raise ValueError('Episode differs')
    if proposal['status']!='completed_proposals_require_accuracy_checks' or not quality['method_unchanged_during_run']:
        raise ValueError('Automatic run incomplete or changed')
    rows=json.loads((episode/'quality/frame-evidence.json').read_text())
    if [r['frame'] for r in rows]!=metadata['frames']:raise ValueError('Frame timeline differs')
    shutil.copytree(source,out,ignore=lambda path,names:['hands'] if Path(path)==source else [])
    frames={side:[] for side in ('left','right')};selected={}
    for row in rows:
        for side,track in [('left','2'),('right','3')]:
            item=row['hands'].get(side,{})
            if not item.get('accepted_by_consistency_proxies',False):continue
            frame=row['frame'];name=f'{frame:06d}'
            pose=episode/'quality/accepted/records'/track/(name+'.json')
            mask=episode/'quality/accepted/masks'/side/(name+'.png')
            r=json.loads(pose.read_text())
            if r['frame_idx']!=frame or bool(r['is_right'])!=(side=='right') or not r['diagnostics']['metric_valid']:
                raise ValueError('Record identity/support differs')
            if r.get('automatic_consistency',{}).get('quality_accepted') is not False:
                raise ValueError('Unexpected approval claim')
            for src,dst in [(pose,out/'hands'/side/'poses'/(name+'.json')),(mask,out/'hands'/side/'masks'/(name+'.png'))]:
                dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
            frames[side].append(frame)
            selected[f'{side}/{name}']={'pose_sha256':sha(pose),'mask_sha256':sha(mask),'geometry_sha256':sha(episode/'quality/geometry'/side/(name+'.npz'))}
    if {s:len(v) for s,v in frames.items()}!=quality['hand_counts']:raise ValueError('Counts differ')
    metadata['hand_frames']=frames;metadata['counts']['hands']={s:len(v) for s,v in frames.items()}
    metadata['sources']['hands_run']='sha256:'+sha(episode/'proposals/receipt.json')
    metadata['hand_recovery']={'method':'frozen automatic current-image observations','proposal_receipt_sha256':sha(episode/'proposals/receipt.json'),'quality_summary_sha256':sha(episode/'quality/summary.json'),'freeze_sha256':proposal['freeze_sha256'],'selected_records':selected,'quality_accepted':False,'pose_accuracy_accepted':False,'fresh_inference_during_staging':False}
    (out/'inputs.json').write_text(json.dumps(metadata,indent=2)+'\n')
    lock={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    (out/'input-files.json').write_text(json.dumps(lock,indent=2)+'\n')
    print(json.dumps({'episode':metadata['episode'],'frames':len(rows),'hand_counts':metadata['counts']['hands'],'files':len(lock),'pose_accuracy_accepted':False}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('kit','episode-root','out'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
