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

R=Path('/workspace/hand-continuity-repair')
D=R/'dex3-isaac-review'


def package(ep, sim_name='sim-v1'):
    label=f'e{ep:03d}'
    root=D/'episodes'/label
    sim=root/sim_name
    receipt=json.loads((sim/'complete.json').read_text())
    target=root/'review'
    if (target/'delivery-hashes.json').exists():
        previous=json.loads((target/'media-verification.json').read_text())
        if previous['comparison.mp4'].get('presentation_timestamps_strictly_increasing') is True:
            return target
        old_archive=D/(label+'-review.tar')
        if old_archive.exists():old_archive.rename(D/(label+'-old-timestamps-'+str(__import__('time').time_ns())+'.tar'))
    if target.exists():
        target.rename(root/('review-incomplete-'+str(__import__('time').time_ns())))
    target.mkdir(exist_ok=False)
    n=receipt['source_frames']
    bundle=json.loads((root/'bundle/manifest.json').read_text())
    fps=bundle['fps']
    if ep in [12,40]:
        human=R/'final-videos'/label/'cad-temporal-hands.mp4'
    else:
        base=R/('all20-retry' if ep==2 else 'all20-results')/label
        if ep in [41,42]:base=R/'roi-recovery'/label/'refit-v1'
        human=base/'videos'/label/'cad-temporal-hands.mp4'
    for name,source in [('human-input.mp4',human),('isaac-source.mp4',sim/'replay.mp4'),('isaac-oblique.mp4',sim/'oblique.mp4'),('isaac-receipt.json',sim/'complete.json'),('retarget-quality.json',root/'retarget/quality.json'),('bundle-validation.json',root/'bundle/validation.json'),('numerical-checks.json',sim/'numerical-checks.json')]:
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
    archive=D/(label+'-review.tar')
    partial=D/(label+'-review.tar.part')
    with tarfile.open(partial,'w') as tar:tar.add(target,arcname=label)
    partial.replace(archive)
    print('REVIEW_PACKAGE_READY',label,n,flush=True)
    return target


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('episode',type=int);p.add_argument('--sim-name',default='sim-v1');a=p.parse_args();package(a.episode,a.sim_name)
