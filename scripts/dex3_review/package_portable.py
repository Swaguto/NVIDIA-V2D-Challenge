"""Produce frame-aligned review media from completed, checked Isaac output."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2




def package(bundle_path, retarget_path, sim, human, target):
    for p in [bundle_path, retarget_path, sim, human, target]:
        if not p.is_absolute():raise ValueError('Use absolute input/output paths')
    receipt=json.loads((sim/'complete.json').read_text())
    bundle=json.loads((bundle_path/'manifest.json').read_text())
    label=f"e{bundle['episode']:03d}"
    contract=json.loads((retarget_path/'manifest.json').read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if receipt['episode']!=bundle['episode'] or receipt['source_frames']!=len(bundle['frames']):
        raise ValueError('Simulator and input episode/frame count differ')
    if receipt['bundle_manifest_sha256']!=sha(bundle_path/'manifest.json'):
        raise ValueError('Simulator is bound to another input bundle')
    if receipt['retarget_manifest_sha256']!=sha(retarget_path/'manifest.json'):
        raise ValueError('Simulator is bound to another retarget result')
    if contract['episode_bundle_manifest_sha256']!=sha(bundle_path/'manifest.json'):
        raise ValueError('Retarget is bound to another input bundle')
    if target.exists():raise FileExistsError('Use a fresh output directory: '+str(target))
    target.mkdir(parents=True)
    n=receipt['source_frames'];fps=bundle['fps']
    for name,source in [('human-input.mp4',human),('isaac-source.mp4',sim/'replay.mp4'),('isaac-oblique.mp4',sim/'oblique.mp4'),('isaac-receipt.json',sim/'complete.json'),('retarget-quality.json',retarget_path/'quality.json'),('bundle-validation.json',bundle_path/'validation.json'),('numerical-checks.json',sim/'numerical-checks.json')]:
        shutil.copyfile(source,target/name)
    assert hashlib.sha256((target/'isaac-source.mp4').read_bytes()).hexdigest()==receipt['video_sha256']
    assert hashlib.sha256((target/'isaac-oblique.mp4').read_bytes()).hexdigest()==receipt['oblique_video_sha256']
    header=Image.new('RGB',(1920,32),(15,21,30));draw=ImageDraw.Draw(header)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
    for x,title in [(12,'Source + repaired human hand estimates'),(652,'Dex3 in Isaac Sim: calibrated source view'),(1292,'Dex3 in Isaac Sim: fixed oblique view')]:
        draw.text((x,5),title,font=font,fill=(232,240,250))
    header.save(target/'comparison-header.png')
    cv2.setNumThreads(2)
    caps=[cv2.VideoCapture(str(p)) for p in [human,sim/'replay.mp4',sim/'oblique.mp4']]
    top=cv2.cvtColor(np.array(header),cv2.COLOR_RGB2BGR)
    encoder=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s','1920x432','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-threads','4','-preset','fast','-crf','19','-pix_fmt','yuv420p',str(target/'comparison.mp4')],stdin=subprocess.PIPE)
    try:
        for fid in range(n):
            panels=[]
            for capture in caps:
                ok,frame=capture.read()
                if not ok:raise ValueError('Input ends before its original frame count')
                panels.append(cv2.resize(frame,(640,400),interpolation=cv2.INTER_AREA))
            encoder.stdin.write(np.vstack([top,np.hstack(panels)]).tobytes())
        if any(capture.read()[0] for capture in caps):raise ValueError('Input has extra frames')
    finally:
        for capture in caps:capture.release()
        encoder.stdin.close()
    if encoder.wait():raise RuntimeError('Comparison encoding failed')
    checks={}
    for p in sorted(target.glob('*.mp4')):
        subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(p),'-f','null','-'],check=True,stdout=subprocess.DEVNULL)
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames,width,height,avg_frame_rate,duration','-of','json',str(p)]))['streams'][0]
        assert int(probe['nb_read_frames'])==n,(label,p.name,probe,n)
        points=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time','-of','json',str(p)]))['frames']
        timestamps=np.array([float(x['best_effort_timestamp_time']) for x in points])
        assert len(timestamps)==n and np.all(np.diff(timestamps)>0), (label,p.name,'nonmonotonic timeline')
        assert abs(timestamps[-1]-(n-1)/fps)<1/fps, (label,p.name,'wrong source duration')
        probe['presentation_timestamps_strictly_increasing']=True
        probe['composition']='source frame index; one decode per input per output frame'
        probe['sha256']=hashlib.sha256(p.read_bytes()).hexdigest();checks[p.name]=probe
    (target/'media-verification.json').write_text(json.dumps(checks,indent=2))
    ids=[round(i*(n-1)/7) for i in range(8)]
    canvas=Image.new('RGB',(1536,4*200),(16,21,29));draw=ImageDraw.Draw(canvas)
    for i,fid in enumerate(ids):
        picture=target/f'sample-{fid:06d}.jpg'
        subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(target/'comparison.mp4'),'-vf',f'select=eq(n\\,{fid}),scale=768:172','-frames:v','1',str(picture)],check=True)
        x=(i%2)*768;y=(i//2)*200
        draw.text((x+8,y+4),f'{label} source frame {fid}',fill=(225,237,250));canvas.paste(Image.open(picture),(x,y+24))
    canvas.save(target/'review-montage.jpg',quality=92)
    shutil.copyfile(target/f'sample-{ids[3]:06d}.jpg',target/'poster.jpg')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in target.iterdir() if p.is_file()}
    (target/'delivery-hashes.json').write_text(json.dumps(hashes,indent=2))
    print('REVIEW_PACKAGE_READY',label,n,flush=True)
    return target


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['bundle','retarget','sim','human','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();package(a.bundle,a.retarget,a.sim,a.human,a.out)
