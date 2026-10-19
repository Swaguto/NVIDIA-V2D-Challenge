"""Locate the retained RGB fit's missing surface ray without changing samples or labels."""
from pathlib import Path
import datetime,hashlib,json,sys
S=Path(__file__).resolve().parents[2];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');sys.path[:0]=[str(S/'mediapipe-cpu-deps-v1'),str(S/'compute')]
import cv2,numpy as np
from numpy_mano_source_geometry_v2 import NumpyMano
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();cv2.setNumThreads(1);O=S/'e037-native-RGB-surface-coverage-v1';O.mkdir()
fit=S/'e037-native-RGB-multiview-source-fit-v1/source-hand-candidate.npz';rp=S/'e037-native-RGB-hand-source-audit-v1/canonical-C-unqualified-record.json';record=json.loads(rp.read_text());request=json.loads((S/'e037-fresh-source-hand-request-v2/request.json').read_text());row=next(x for x in request['frames'] if x['frame_idx']==50)
with np.load(request['canonical_bundle']) as z:K=z['intrinsics'].copy()
with np.load(fit) as z:
 pose=z['mano_pose'][0].copy();t=z['camera_translation'][0].copy();px=z['heldout_pixels'].copy();front=z['heldout_front_depth_m'].copy();depth=z['heldout_estimated_source_depth_m'].copy();train=z['train_pixels'].copy()
m=NumpyMano(S/'models/mano/models/MANO_RIGHT.pkl');v,j,_=m.record(record,pose=pose,translation=t)
uv=np.c_[K[0,0]*v[:,0]/v[:,2]+K[0,2],K[1,1]*v[:,1]/v[:,2]+K[1,2]]
image=cv2.imread(str(S/'e037-fresh-source-hand-request-v1/images/000050.png'));mask=np.zeros(image.shape[:2],np.uint8)
for triangle in m.faces:cv2.fillConvexPoly(mask,np.rint(uv[triangle]).astype(int),255)
rows=[]
for i,p in enumerate(px):
 xy=tuple(map(int,p));miss=not np.isfinite(front[i]);distance=float(np.sqrt(np.square(uv-p).sum(1)).min());rows.append(dict(index=i,source_pixel=p.tolist(),heldout_ray_hit=not miss,estimated_source_skin_depth_m=float(depth[i]),estimated_model_front_depth_m=float(front[i]) if not miss else None,nearest_projected_vertex_distance_px=distance,rasterized_estimated_mesh_silhouette_contains_pixel=bool(mask[xy[1],xy[0]])))
 colour=(0,0,255) if miss else (0,220,0);cv2.circle(image,xy,3,colour,-1)
 if miss:cv2.putText(image,str(i),(xy[0]+5,xy[1]),cv2.FONT_HERSHEY_SIMPLEX,.5,colour,1)
contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(image,contours,-1,(255,180,0),1)
crop=row['crop'];lo=np.floor(np.array([crop['cx'],crop['cy']])-crop['size']/2).astype(int);hi=np.ceil(np.array([crop['cx'],crop['cy']])+crop['size']/2).astype(int);cv2.imwrite(str(O/'heldout-surface-coverage.png'),cv2.resize(image[lo[1]:hi[1],lo[0]:hi[0]],(768,768)))
report=dict(schema='v2d.native-RGB-fit-heldout-surface-coverage.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),candidate_sha256=sha(fit),record_sha256=sha(rp),samples_or_gate_modified=False,source_skin_labels_are_inferred=True,estimated_model_visibility_is_not_video_visibility=True,physical_task_accepted=False,rows=rows)
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps([x for x in rows if not x['heldout_ray_hit']],indent=2))
