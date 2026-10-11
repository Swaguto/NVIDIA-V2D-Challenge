"""Stage a separate hand-recovery ablation; never overwrite a reference pack.

This imports saved metric-valid camera-frame fits. It does not claim fresh
inference, approve hand identity, or interpolate missing observations.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def contains_interpolation(value):
    if isinstance(value,dict):
        return value.get('interpolated') is True or any(contains_interpolation(v) for v in value.values())
    if isinstance(value,list):return any(contains_interpolation(v) for v in value)
    return False

def main(args):
    kit=args.kit.resolve();out=args.out.resolve()
    if out.exists():raise FileExistsError(out)
    if sha(args.archive)!=args.sha256:raise ValueError('Recovery archive hash differs')
    manifest=json.loads((kit/'artifact-manifest.json').read_text())
    for name,row in manifest['files'].items():
        if sha(kit/name)!=row['sha256']:raise ValueError('Changed baseline file: '+name)
    source=kit/'perception-checkpoint'
    metadata=json.loads((source/'inputs.json').read_text())
    frames=set(metadata['frames']);episode=metadata['episode']
    records={'left':{},'right':{}};masks={};evidence={}
    prefix=f'reconstruction/runs/e{episode:03d}/'
    with tarfile.open(args.archive,'r|gz') as tar:
        for member in tar:
            if not member.isfile():continue
            if member.name.startswith(prefix+'align/records/') and member.name.endswith('.json'):
                raw=tar.extractfile(member).read();r=json.loads(raw)
                frame=int(r['frame_idx']);side='right' if r['is_right'] else 'left'
                if frame not in frames:raise ValueError('Unexpected frame')
                if r.get('diagnostics',{}).get('metric_valid') is not True:continue
                if r.get('intrinsics',{}).get('id')!='ego_cam_c':raise ValueError('Camera mismatch')
                if r.get('directly_observed') is not True:raise ValueError('Missing direct observation provenance')
                if contains_interpolation(r.get('direct_observation',{})):raise ValueError('Interpolated input')
                if frame in records[side]:raise ValueError('Ambiguous duplicate hand fit')
                records[side][frame]=raw
                evidence[f'{side}/{frame:06d}']={'archive_member':member.name,'sha256':hashlib.sha256(raw).hexdigest(),
                    'kind':r.get('direct_observation',{}).get('kind','unknown')}
            elif member.name.startswith(prefix+'hands/masks/') and member.name.endswith('.png'):
                rel=PurePosixPath(member.name[len(prefix+'hands/masks/'):])
                if len(rel.parts)!=2 or rel.parts[0] not in records or int(rel.stem) not in frames:
                    raise ValueError('Unexpected mask path')
                masks[rel.as_posix()]=tar.extractfile(member).read()
    for side,rows in records.items():
        if not rows:raise ValueError('No supported '+side+' poses')
        for frame in rows:
            if f'{side}/{frame:06d}.png' not in masks:raise ValueError('Missing mask')
    # Independent experiment: retain the baseline scene/depth/camera inputs.
    shutil.copytree(source,out,ignore=lambda p,n: ['hands'] if Path(p)==source else [])
    for side,rows in records.items():
        for frame,raw in rows.items():
            p=out/'hands'/side/'poses'/f'{frame:06d}.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
        for rel,raw in masks.items():
            if rel.startswith(side+'/'):
                p=out/'hands'/side/'masks'/PurePosixPath(rel).name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    metadata['hand_frames']={s:sorted(rows) for s,rows in records.items()}
    metadata['counts']['hands']={s:len(rows) for s,rows in records.items()}
    metadata['sources']['hands_run']='sha256:'+args.sha256
    metadata['hand_recovery']={'archive_sha256':args.sha256,'fresh_inference':False,
        'quality_accepted':False,'scope':'saved hand fits and masks only; other baseline inputs preserved',
        'selected_records':evidence}
    (out/'inputs.json').write_text(json.dumps(metadata,indent=2)+'\n')
    lock={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    (out/'input-files.json').write_text(json.dumps(lock,indent=2)+'\n')
    print(json.dumps({'episode':episode,'counts':metadata['counts']['hands'],'files':len(lock),'quality_accepted':False}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('kit','archive','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--sha256',required=True)
    main(p.parse_args())
