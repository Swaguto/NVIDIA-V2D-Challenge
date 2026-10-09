"""GT-free, fixed-translation/axis-tilt spin sweep of the textured lid CAD."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import sys

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--repo',type=Path,required=True)
p.add_argument('--data',type=Path,required=True)
p.add_argument('--observations',type=Path,required=True)
p.add_argument('--masks',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
args=p.parse_args()
if args.out.exists():p.error('fresh output required')
os.environ['V2D_PUBLIC']=str(args.data)
os.environ['V2D_CALIB']=str(args.data/'meta/camera_calibration.json')
sys.path.insert(0,str(args.repo/'scripts/track3'))
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import register_poses as reg
from v2d.world_calib.vm.fit_ego_rig import load_mesh

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
raw=args.observations.read_bytes();baseline=json.loads(raw)
if (baseline['gt_use']!='none' or baseline['coordinate_frame']!='rectified_camera'
        or baseline['camera_frame']!='ego_cam_c'):
    raise ValueError('GT-free rectified-left observations required')
for item in baseline['inputs']:
    if sha(args.data/item['path'])!=item['sha256']:raise ValueError('RGB/CAD input changed')
if sha(args.data/'meta/camera_calibration.json')!=baseline['calibration_sha256']:
    raise ValueError('calibration changed')
rig=reg.load_rig()
mesh=load_mesh(args.data/'mesh/white_pot_lid/white_pot_lid.glb')
vertices=np.asarray(mesh.vertices);faces=np.asarray(mesh.faces)
colors=np.asarray(mesh.visual.to_color().vertex_colors)[:,:3]
values,vectors=np.linalg.eigh(np.cov(vertices.T))
axis=vectors[:,0]
if axis[0]<0:axis=-axis
tangent1=np.array([0.,1.,0.]);tangent1-=np.dot(tangent1,axis)*axis;tangent1/=np.linalg.norm(tangent1)
tangent2=np.cross(axis,tangent1)
planar=np.column_stack([vertices@tangent1,vertices@tangent2])
height=vertices@axis
size=512;border=24
scale=(size-2*border)/np.ptp(planar,axis=0).max()
offset=planar.min(0)-border/scale
uv=np.rint((planar-offset)*scale).astype('int32')
template=np.zeros((size,size,3),np.uint8)
hit=np.zeros((size,size),np.uint8)
height_image=np.full((size,size),np.nan,np.float32)
for face in faces[np.argsort(height[faces].mean(1))]:
    rgb=tuple(int(x) for x in np.mean(colors[face],axis=0))
    cv2.fillConvexPoly(template,uv[face],rgb)
    cv2.fillConvexPoly(hit,uv[face],1)
    cv2.fillConvexPoly(height_image,uv[face],float(height[face].mean()))
yy,xx=np.indices(hit.shape)
physical=np.stack([xx/scale+offset[0],yy/scale+offset[1]],axis=-1)
radius=np.linalg.norm(physical,axis=-1)
roi=(hit>0)&(radius>.035)&(radius<.086)
plane_height=float(np.nanmedian(height_image[roi]))
origin=plane_height*axis+offset[0]*tangent1+offset[1]*tangent2
gray=cv2.cvtColor(template,cv2.COLOR_RGB2GRAY).astype('float32')
args.out.mkdir(parents=True)
cv2.imwrite(str(args.out/'canonical_texture.png'),cv2.cvtColor(template,cv2.COLOR_RGB2BGR))
cv2.imwrite(str(args.out/'canonical_roi.png'),roi.astype('uint8')*255)
np.savez_compressed(args.out/'canonical_template.npz',gray=gray,roi=roi,axis=axis,
                   tangent1=tangent1,tangent2=tangent2,origin=origin,scale=scale)
report=copy.deepcopy(baseline)
report.update(frames=[],producer='fixed-axis textured-lid spin appearance diagnostic',
              observation_input_sha256=hashlib.sha256(raw).hexdigest(),
              worker_sha256_lf=hashlib.sha256(Path(__file__).read_text().encode()).hexdigest())
report['params']['lid_spin']={'axis_in_cad':axis.tolist(),'covariance_eigenvalues':values.tolist(),
    'plane_height_m':plane_height,'texture_roi_radius_m':[.035,.086],
    'coarse_step_deg':5,'fine_step_deg':.5,'translation_fixed':True,'axis_tilt_fixed':True,
    'score':'normalized high-pass grayscale correlation','highpass_sigma_px':3,
    'gt_use':'none','template_sha256':sha(args.out/'canonical_template.npz')}
for row0 in baseline['frames']:
    ep,frame=row0['episode'],row0['frame_index']
    row=copy.deepcopy(row0);fit=row['fits']['white_pot_lid']
    R0=Rotation.from_quat(np.array(fit['rotation_wxyz'])[[1,2,3,0]]).as_matrix()
    t=np.array(fit['translation'])
    image=rig.rectify_pair(*[reg.read_frame(args.data/f'videos/chunk-000/observation.images.{cam}/episode_{ep:06d}.mp4',frame) for cam in (rig.left,rig.right)])[0]
    mp=args.masks/f'e{ep:03d}/{frame:06d}/white_pot_lid_mask.png'
    if sha(mp)!=row['views'][rig.left]['objects']['white_pot_lid']['mask_sha256']:raise ValueError('mask changed')
    mask=cv2.imread(str(mp),0)>0
    yy,xx=np.nonzero(mask)
    x0=max(int(xx.min())-10,0);x1=min(int(xx.max())+11,image.shape[1])
    y0=max(int(yy.min())-10,0);y1=min(int(yy.max())+11,image.shape[0])
    observed=cv2.cvtColor(image[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY).astype('float32')
    observed-=cv2.GaussianBlur(observed,(0,0),3)
    target=mask[y0:y1,x0:x1]
    K=rig.P1[:3,:3].copy();K[0,2]-=x0;K[1,2]-=y0
    def cost(theta):
        R=R0@Rotation.from_rotvec(axis*np.deg2rad(theta)).as_matrix()
        H=K@np.column_stack([R@tangent1/scale,R@tangent2/scale,R@origin+t])
        rendered=cv2.warpPerspective(gray,H,(x1-x0,y1-y0))
        eligible=(cv2.warpPerspective(roi.astype('float32'),H,(x1-x0,y1-y0))>.99)&target
        rendered-=cv2.GaussianBlur(rendered,(0,0),3)
        a,b=observed[eligible],rendered[eligible]
        if len(a)<100:return {'angle_deg':float(theta),'cost':2.,'pixels':len(a)},R,H
        a=a-a.mean();b=b-b.mean()
        ncc=float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-9))
        return {'angle_deg':float(theta),'cost':1-ncc,'pixels':len(a)},R,H
    coarse=[cost(a)[0] for a in np.arange(0,360,5)]
    seed=min(coarse,key=lambda c:c['cost'])['angle_deg']
    fine=[cost(a)[0] for a in np.arange(seed-5,seed+5.01,.5)]
    best=min(fine,key=lambda c:c['cost'])
    _,R,H=cost(best['angle_deg'])
    separated=[c for c in coarse if abs((c['angle_deg']-best['angle_deg']+180)%360-180)>=30]
    alternative=min(separated,key=lambda c:c['cost'])
    fit['rotation_wxyz']=reg._mat_to_wxyz(R).tolist()
    fit['appearance_spin_deg']=best['angle_deg']%360
    row['lid_spin_diagnostic']={'coarse_curve':coarse,'fine_curve':fine,'selected':best,
        'alternative_at_least30deg':alternative,'separated_peak_cost_gap':alternative['cost']-best['cost'],
        'baseline_cost':coarse[0]['cost'],'translation_fixed':True,'axis_tilt_fixed':True,
        'post_spin_geometric_residual':'not recomputed; inherited residual is baseline only'}
    # The inherited geometric residual is not a validity gate on the spun pose.
    fit['baseline_geometric_residual_m']=fit.pop('residual_m')
    warped=cv2.warpPerspective(gray,H,(x1-x0,y1-y0))
    display=np.concatenate([image[y0:y1,x0:x1],cv2.cvtColor(np.uint8(np.clip(warped,0,255)),cv2.COLOR_GRAY2BGR)],axis=1)
    cv2.imwrite(str(args.out/f'{frame:06d}_appearance.jpg'),display)
    report['frames'].append(row)
    (args.out/'probe_results.partial.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'frame':frame,'spin':best,'baseline_cost':coarse[0]['cost'],'gap':alternative['cost']-best['cost']}),flush=True)
(args.out/'probe_results.json').write_text(json.dumps(report,indent=2))
if args.observations.read_bytes()!=raw:raise ValueError('predictions changed')
print('Frozen appearance spin diagnostic, no pose GT read.',flush=True)
