"""Make source-to-simulator review video and exhaustive frame contact sheets."""
import argparse
import json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def main(a):
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    receipt=json.loads((a.replay/'complete.json').read_text())
    episode=json.loads((a.inputs/'inputs.json').read_text())['episode']
    audit=json.loads((a.replay/'independent-audit.json').read_text())
    assert audit['frames']==receipt['decoded_video_frames']
    with np.load(a.replay/'simulator-readback.npz') as readback:
        frames=readback['frame_indices']; supported=readback['robot_valid']; pixels=readback['hand_review_color_pixels']
    try:
        font=ImageFont.truetype('DejaVuSans.ttf',22)
        small=ImageFont.truetype('DejaVuSans.ttf',14)
    except OSError:
        font=small=ImageFont.load_default()
    writer=imageio.get_writer(a.out/'source-to-dex3-continuous.mp4',fps=receipt['fps'],codec='libx264',quality=8,macro_block_size=1)
    sheet=None
    try:
        for row,frame in enumerate(frames):
            rgb=Image.open(a.replay/'frames'/f'{int(frame):06d}.png').convert('RGB')
            source=Image.open(a.inputs/'frames'/f'{int(frame):06d}.png').convert('RGB')
            canvas=Image.new('RGB',(1920,832),(20,23,28)); draw=ImageDraw.Draw(canvas)
            draw.text((20,12),f'e{episode:03d} | frame {frame} | source video -> actual Isaac Sim 6 / Dex3 replay',font=font,fill='white')
            draw.text((20,48),'SOURCE IMAGE',font=font,fill='white')
            draw.text((980,48),'DEX3: left blue / right orange | kinematic playback',font=font,fill='white')
            source.thumbnail((960,720)); canvas.paste(source,((960-source.width)//2,80+(720-source.height)//2))
            canvas.paste(rgb.resize((960,720),Image.Resampling.LANCZOS),(960,80))
            state=' | '.join(f'{side}: '+('source fit' if supported[row,s] else 'HELD, unobserved') for s,side in enumerate(('Left','Right')))
            draw.text((20,802),state+' | Hand placement accuracy UNAPPROVED',font=font,fill=(255,200,80))
            writer.append_data(np.asarray(canvas))
            # Every frame, including all held intervals, is included in sheets.
            if row%30==0:sheet=Image.new('RGB',(1920,5*270),(20,23,28))
            x=(row%6)*320; y=((row%30)//6)*270
            sheet.paste(rgb.resize((320,240),Image.Resampling.LANCZOS),(x,y))
            label=f'{int(frame):03d} L:{"fit" if supported[row,0] else "HOLD"} R:{"fit" if supported[row,1] else "HOLD"}'
            ImageDraw.Draw(sheet).text((x+5,y+242),label,font=small,fill='white')
            if row%30==29 or row==len(frames)-1:sheet.save(a.out/f'contact-sheet-{row//30+1:02d}.jpg',quality=94)
    finally:writer.close()
    with imageio.get_reader(a.out/'source-to-dex3-continuous.mp4') as reader:
        assert reader.count_frames()==len(frames)
    (a.out/'review-index.json').write_text(json.dumps(dict(frames=len(frames),
        contact_sheets=(len(frames)+29)//30,minimum_pixel_frames=[int(frames[np.argmin(pixels[:,s])]) for s in range(2)],
        largest_pixel_drop_frames=[int(frames[1+np.argmin(np.diff(pixels[:,s]))]) for s in range(2)],
        scope='Exhaustive model-presence review; not hand accuracy ground truth'),indent=2)+'\n')
    print(a.out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('inputs','replay','out'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
