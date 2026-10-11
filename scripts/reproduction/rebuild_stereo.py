"""Recompute an episode's stereo sequence; optionally verify frozen array hashes."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import cv2
import numpy as np
from stereo_depth import load_rig,stereo_depth

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main(args):
    if args.out.exists():raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    calibration=args.data/'meta/camera_calibration.json'
    rig=load_rig(str(calibration))
    inputs=[calibration]+[args.data/f'videos/chunk-{args.chunk:03d}/observation.images.{cam}/episode_{args.episode:06d}.mp4' for cam in (rig.left,rig.right)]
    caps=[cv2.VideoCapture(str(p)) for p in inputs[1:]]
    if not all(c.isOpened() for c in caps):raise ValueError('Stereo videos unavailable')
    source_hashes={p.relative_to(args.data).as_posix():sha(p) for p in inputs}
    expected=json.loads(args.reference_receipt.read_text()) if args.reference_receipt else None
    if expected and (expected['episode']!=args.episode or expected['input_sha256']!=source_hashes):
        raise ValueError('Reference episode or input hashes differ')
    for name in ('color','depth','valid'):(args.out/name).mkdir()
    rows=[];start=time.monotonic()
    i=0
    while True:
        frames=[c.read() for c in caps]
        if not any(x[0] for x in frames):break
        if not all(x[0] for x in frames):raise ValueError(f'Early video end: {i}')
        left,right=[x[1] for x in frames]
        color,_=rig.rectify_pair(left,right)
        depth,valid=stereo_depth(rig,left,right)
        depth=np.where(valid,depth,np.nan)
        stem=f'{i:06d}'
        cv2.imwrite(str(args.out/f'color/{stem}.png'),color)
        cv2.imwrite(str(args.out/f'valid/{stem}.png'),valid.astype(np.uint8)*255)
        np.save(args.out/f'depth/{stem}.npy',depth)
        record={'frame':i,'valid_depth_pixels':int(valid.sum()),
                'color_array_sha256':hashlib.sha256(color.tobytes()).hexdigest(),
                'depth_array_sha256':hashlib.sha256(depth.tobytes()).hexdigest(),
                'depth_shape':list(depth.shape),'depth_dtype':str(depth.dtype)}
        if args.reference:
            episode=f'e{args.episode:03d}'
            color_path=args.reference/f'issue6-full-inputs/{episode}/color/{stem}.png'
            depth_path=args.reference/f'issue6-full-runs/all20-final-v3/{episode}/depth/{stem}.npy'
            old_color=cv2.imread(str(color_path))
            old_depth=np.load(depth_path)
            shared=np.isfinite(old_depth)&np.isfinite(depth)
            record.update(color_exact=bool(np.array_equal(old_color,color)),
                depth_nan_mask_exact=bool(np.array_equal(np.isnan(depth),np.isnan(old_depth))),
                depth_exact=bool(np.array_equal(depth,old_depth,equal_nan=True)),
                depth_max_delta_m=float(np.max(np.abs(old_depth[shared]-depth[shared]))) if shared.any() else None)
        if expected:
            if i>=len(expected['rows']):raise ValueError('Extra frames relative to receipt')
            old=expected['rows'][i]
            record.update(color_exact=record['color_array_sha256']==old['color_array_sha256'],
                depth_exact=all(record[k]==old[k] for k in ('depth_array_sha256','depth_shape','depth_dtype')))
        rows.append(record)
        if i%30==0:print(json.dumps(record),flush=True)
        i+=1
    if not rows:raise ValueError('No decoded stereo frames')
    if args.expected_frames is not None and len(rows)!=args.expected_frames:raise ValueError('Frame count differs')
    if expected and len(rows)!=expected['frames']:raise ValueError('Reference frame count differs')
    for c in caps:c.release()
    for p in inputs:
        if sha(p)!=source_hashes[p.relative_to(args.data).as_posix()]:raise ValueError('Input changed')
    report={'episode':args.episode,'frames':len(rows),'opencv':cv2.__version__,'numpy':np.__version__,
        'input_sha256':source_hashes,'source_sha256':sha(Path(__file__).with_name('stereo_depth.py')),
        'elapsed_s':time.monotonic()-start,'fresh_raw_video_stereo':True,'rows':rows}
    if args.reference or expected:
        report.update(all_colors_exact=all(r['color_exact'] for r in rows),
                      all_depth_exact=all(r['depth_exact'] for r in rows))
    (args.out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    if (args.reference or expected) and not (report['all_colors_exact'] and report['all_depth_exact']):
        raise ValueError('Stereo reconstruction differs from reference; inspect receipt.json')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--episode',type=int,required=True)
    p.add_argument('--chunk',type=int,default=0)
    p.add_argument('--expected-frames',type=int)
    group=p.add_mutually_exclusive_group()
    group.add_argument('--reference',type=Path,help='Historical episode cache')
    group.add_argument('--reference-receipt',type=Path,help='Portable reference receipt with array hashes')
    main(p.parse_args())
