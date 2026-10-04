"""RGB-prompted SAM2 observation control; no object-pose GT inputs."""
from pathlib import Path
import copy, hashlib, json, os, sys, time
import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data',type=Path,required=True)
parser.add_argument('--models',type=Path,required=True)
parser.add_argument('--baseline',type=Path,required=True,help='native pot probe output directory')
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--annotations',type=Path,help='hash-bound RGB-only prompts; default preserves the historical episode-2 control')
args = parser.parse_args()
REPO = Path(__file__).resolve().parents[2]
DATA, MODELS, BASE, OUT = args.data, args.models, args.baseline, args.out
os.environ['V2D_PUBLIC'] = str(DATA)
os.environ['V2D_CALIB'] = str(DATA / 'meta/camera_calibration.json')
os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, str(REPO / 'scripts/track3'))
import cv2, numpy as np, torch, transformers
from transformers import Sam2Model, Sam2Processor
import register_poses as reg
from v2d.world_calib.vm.fit_ego_rig import load_mesh
from v2d.world_calib.vm.visualize_object_pose import render_depth
from rgb_annotations import load_rgb_annotations, verify_review_image

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
if OUT.exists(): raise ValueError('fresh output required')
baseline = json.loads((BASE/'probe_results.json').read_text())
if baseline['gt_use'] != 'none': raise ValueError('GT-assisted baseline')
if sha(DATA/'meta/camera_calibration.json') != baseline['calibration_sha256']: raise ValueError('changed calibration')
rig = reg.load_rig()
checkpoint = MODELS / 'facebook--sam2.1-hiera-large/665f8e2ad61cf5f53d65644ff27c8ee525124610'
for item in baseline['models']:
    p = MODELS / item['model'].replace('/','--') / item['revision'] / item['file']
    if sha(p) != item['sha256']: raise ValueError('changed model input')
for item in baseline['inputs']:
    if sha(DATA/item['path']) != item['sha256']: raise ValueError('changed data input')
names = baseline['objects']
if names != ['white_pot', 'white_pot_lid']:
    raise ValueError('expected pot/lid object order')
# Coordinates reviewed from rectified RGB only, before any pose evaluation.
# Each tuple: box, foreground points, background/other-object/occluder points.
annotations = {
 0: [[[517,420,730,800],[[630,611],[647,730]],[[625,520],[790,650]]],
     [[517,420,730,600],[[620,508]],[[644,630],[790,600]]]],
 49: [[[534,406,736,800],[[610,598],[676,749]],[[622,490],[643,676],[790,650]]],
      [[534,406,729,583],[[591,506]],[[658,536],[634,620]]]],
 98: [[[487,415,705,800],[[596,554],[619,770]],[[560,735],[850,555],[740,620]]],
      [[740,458,947,640],[[818,539]],[[882,556],[760,651]]]],
 147: [[[456,350,682,767],[[560,498],[619,716],[583,558]],[[555,665],[860,549],[700,625]]],
       [[773,459,990,634],[[864,551]],[[744,595],[1000,656]]]],
 195: [[[464,375,681,800],[[565,526],[589,767],[581,590]],[[553,693],[862,540],[730,630]]],
       [[763,459,979,634],[[858,546]],[[742,599],[1000,658]]]],
 244: [[[459,391,680,800],[[570,553],[583,729]],[[720,674],[860,530]]],
       [[751,447,952,619],[[817,536]],[[885,569],[724,640]]]],
 293: [[[531,394,746,800],[[632,591],[663,766]],[[622,485],[642,651],[800,635]]],
       [[531,394,741,568],[[585,482]],[[656,509],[644,600]]]],
 342: [[[521,401,729,800],[[631,584],[646,720]],[[626,501],[791,647]]],
       [[521,401,729,575],[[622,485]],[[643,615],[790,600]]]],
}
episode, reviewed_images = 2, {}
if args.annotations:
    episode, annotations, reviewed_images = load_rgb_annotations(
        args.annotations, baseline, sha(BASE/'probe_results.json'))
torch.set_num_threads(4)
processor = Sam2Processor.from_pretrained(checkpoint)
model = Sam2Model.from_pretrained(checkpoint).to('cuda').eval()
cads = {n: torch.tensor(reg.load_cad(n,6000,0),device='cuda',dtype=torch.float32) for n in names}
meshes = {n: load_mesh(DATA/f'mesh/{n}/{n}.glb') for n in names}
OUT.mkdir(parents=True)
report=copy.deepcopy(baseline)
report.update(episodes=[episode],frames=[],baseline_sha256=sha(BASE/'probe_results.json'),
              gt_use='none',prompt=None,producer='RGB reviewed per-frame SAM2 box/point prompts; provisional control',
              coordinate_frame='rectified_camera',camera_frame=rig.left,
              annotations=annotations,worker_sha256_lf=hashlib.sha256(Path(__file__).read_text().encode()).hexdigest(),
              limits=[f'8 preselected episode-{episode} frames, rectified ego_cam_c only; not continuous tracking.',
                      'Prompts reviewed by Codex from RGB; no independent human mask review or mask-IoU truth.',
                      'Depth/CAD/native solver/gate unchanged; no GT read before prediction freeze.',
                      'Lid-mask pixels are excluded from the pot mask; hands and other uncertain pixels may remain.',
                      'All masks require independent review; no automatic acquisition fix is claimed.',
                      'FoundationPose is not run; camera-space fitting does not establish world-pose accuracy.'])
if args.annotations:
    report['annotations_input_sha256'] = sha(args.annotations)
    report['annotations_source_sha256_lf'] = hashlib.sha256((REPO/'scripts/track3/rgb_annotations.py').read_text().encode()).hexdigest()
for original in baseline['frames']:
    if original['episode'] != episode: continue
    frame=original['frame_index']
    folder=OUT/f'e{episode:03d}/{frame:06d}';folder.mkdir(parents=True)
    raw=[reg.read_frame(DATA/f'videos/chunk-000/observation.images.{cam}/episode_{episode:06d}.mp4',frame) for cam in [rig.left,rig.right]]
    image=rig.rectify_pair(*raw)[0]
    if frame in reviewed_images:
        verify_review_image(image,reviewed_images[frame])
    rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    frozen=BASE/f'e{episode:03d}/{frame:06d}/depth.npz'
    if sha(frozen)!=original['depth_sha256']: raise ValueError('depth changed')
    observation=np.load(frozen);depth,valid=observation['depth'],observation['valid']
    points=reg.backproject_full(depth,valid,rig.P1)
    row={'episode':episode,'frame_index':frame,'total_source_frames':original['total_source_frames'],
         'depth_sha256':original['depth_sha256'],'views':{rig.left:{'objects':{}}},'fits':{}}
    overlay=image.copy()
    masks={}
    for i,name in enumerate(names):
        box,positive,negative=annotations[frame][i]
        inputs=processor(images=rgb,input_boxes=[[box]],input_points=[[positive+negative]],
                         input_labels=[[[1]*len(positive)+[0]*len(negative)]],return_tensors='pt').to('cuda')
        with torch.inference_mode(): output=model(**inputs,multimask_output=False)
        kw={}
        if inputs.get('reshaped_input_sizes') is not None: kw['reshaped_input_sizes']=inputs['reshaped_input_sizes']
        mask=processor.post_process_masks(output.pred_masks,inputs['original_sizes'],binarize=True,**kw)[0][0,0].cpu().numpy().astype(bool)
        masks[name]=mask
        path=folder/f'{name}_prompted_mask.png';cv2.imwrite(str(path),mask.astype('uint8')*255)
    # On stacked frames the lid is an external occluder, excluded from pot depth.
    # This is a measured observation control, not a general production union rule.
    row['pot_lid_overlap_pixels_before_exclusion']=int((masks[names[0]]&masks[names[1]]).sum())
    masks[names[0]]=masks[names[0]]&~masks[names[1]]
    for i,name in enumerate(names):
        mask=masks[name]
        path=folder/f'{name}_mask.png';cv2.imwrite(str(path),mask.astype('uint8')*255)
        colors=[np.array([0,255,0]),np.array([255,0,255])]
        overlay[mask]=np.uint8(.65*overlay[mask]+.35*colors[i])
        selected=mask.ravel()&valid.ravel()
        detail={'mask_pixels':int(mask.sum()),'valid_depth_pixels':int(selected.sum()),'mask_sha256':sha(path)}
        row['views'][rig.left]['objects'][name]=detail
        fit={'status':'rejected','reason':'insufficient_masked_depth'};row['fits'][name]=fit
        if selected.sum()<300: continue
        cloud=reg.trim_cloud(reg.voxel_downsample(points[selected],.004),method='radial')
        fit['cloud_points']=len(cloud)
        if len(cloud)<120: continue
        s,R,t,res=reg.register(cads[name],torch.tensor(cloud,device='cuda',dtype=torch.float32))
        fit.update(scale=float(s),rotation_wxyz=reg._mat_to_wxyz(R.cpu().numpy()).tolist(),
                   translation=t.cpu().numpy().tolist(),residual_m=float(res),
                   status='observed' if np.isfinite(res) and res<=.05 else 'rejected',reason=None if res<=.05 else 'residual_gate')
        mesh=meshes[name];vertices=np.asarray(mesh.vertices)@R.cpu().numpy().T+t.cpu().numpy()
        rendered=render_depth(vertices,mesh.faces,rig.P1[:3,:3],image.shape[:2]);sil=np.isfinite(rendered)&(rendered>0)
        fit['cad_model_mask_iou']=float((mask&sil).sum()/max(1,(mask|sil).sum()))
        contours=cv2.findContours(sil.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(overlay,contours,-1,(0,255,255),2)
    cv2.putText(overlay,f'ep{episode} {frame}: prompted pot green / lid magenta / CAD yellow',(10,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
    cv2.imwrite(str(folder/'overlay.jpg'),overlay)
    report['frames'].append(row)
    print(json.dumps({'frame':frame,'fits':row['fits'],'masks':row['views'][rig.left]['objects']}),flush=True)
    (OUT/'probe_results.partial.json').write_text(json.dumps(report,indent=2))
(OUT/'probe_results.json').write_text(json.dumps(report,indent=2))
print('Frozen reviewed prompt predictions',flush=True)
