"""Compare saved baseline meshes using the same image/depth consistency checks.

Masks come from the candidate, not ground-truth annotations. Baseline and new
fits have different alignment histories. This is a diagnostic, not pose accuracy.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import cv2
import numpy as np
import pyrender
import trimesh
from automatic_hand_quality import classify
from automatic_hands import sha,write

p=argparse.ArgumentParser(description=__doc__)
for name in ('source','depth-root','baseline','quality','mano-assets','out'):
    p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--config',type=Path,default=Path(__file__).with_name('automatic_hands.json'))
a=p.parse_args()
if a.out.exists():raise FileExistsError(a.out)
module_path=Path(__file__).parents[2]/'modules/v2d_hamer/lib/render_hands_aligned_video.py'
spec=importlib.util.spec_from_file_location('baseline_hand_renderer',module_path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
mano=module._mano_layer(str(a.mano_assets))
manifest=json.loads((a.source/'manifest.json').read_text());c=manifest['color_camera']
policy=json.loads(a.config.read_text());evidence=json.loads((a.quality/'frame-evidence.json').read_text())
camera=pyrender.IntrinsicsCamera(c['fx'],c['fy'],c['cx'],c['cy'])
renderer=pyrender.OffscreenRenderer(c['width'],c['height']);rows=[];baseline_support={s:[] for s in ('left','right')}
try:
    for row in evidence:
        frame=row['frame']
        depth=np.load(a.depth_root/f'{frame:06d}.npy',allow_pickle=False)
        validity=a.source/'valid'/f'{frame:06d}.png'
        if validity.exists():depth=np.where(cv2.imread(str(validity),0)>0,depth,np.nan)
        for side,track in [('left','2'),('right','3')]:
            record_path=a.baseline/'records'/track/f'{frame:06d}.json'
            if not record_path.exists():continue
            record=json.loads(record_path.read_text())
            if not record.get('diagnostics',{}).get('metric_valid',False):continue
            baseline_support[side].append(frame)
            item=row['hands'].get(side)
            if item is None:continue
            mask=cv2.imread(str(a.quality/'masks'/side/f'{frame:06d}.png'),0)>0
            vertices,faces=module._build_hand_mesh(record,mano)
            scene=pyrender.Scene();scene.add(camera,pose=np.eye(4))
            mesh=trimesh.Trimesh(vertices*[1,-1,-1],faces,process=False)
            scene.add(pyrender.Mesh.from_trimesh(mesh,smooth=False))
            rendered=renderer.render(scene,flags=pyrender.RenderFlags.DEPTH_ONLY|pyrender.RenderFlags.SKIP_CULL_FACES)
            measured=classify(mask,rendered,depth,item['mask_score'],policy)
            rows.append({'frame':frame,'side':side,'baseline':measured,
                         'baseline_consistent':not measured['reasons'],
                         'automatic_consistent':not [x for x in item['reasons'] if x!='pose_jump_inconsistent_with_image_motion'],
                         'baseline_record_sha256':sha(record_path)})
finally:renderer.delete()
summary={s:{'baseline_supported_frames':len(baseline_support[s]),
            'common_frames':sum(r['side']==s for r in rows),
            'baseline_consistent':sum(r['side']==s and r['baseline_consistent'] for r in rows),
            'automatic_consistent':sum(r['side']==s and r['automatic_consistent'] for r in rows)} for s in ('left','right')}
write(a.out,{'scope':__doc__,'episode':manifest['episode'],'config_sha256':sha(a.config),
             'quality_evidence_sha256':sha(a.quality/'frame-evidence.json'),'summary':summary,
             'baseline_supported_frame_ids':baseline_support,'rows':rows})
print(json.dumps(summary))
