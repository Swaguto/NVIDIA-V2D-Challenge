"""Independent explicit chain/quaternion/Rodrigues check of NEW branch returns.

Does not call the producer, shared FK, native runtime or any old source audit.
"""
from pathlib import Path
import datetime, hashlib, json, math, os, sys, time
import xml.etree.ElementTree as ET
import numpy as np
S=Path('/Users/rohithiruku/v2d_scored_baseline_20261017');O=S/'e037-preserved-branch-independent-v1';P=S/'e037-preserved-branch-source-v1';F=S/'e037-thumb-ray-independent-validation-v1/frozen';start=time.monotonic()
from thumb_conic_geometry_v1 import plain,journal
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def save(n,v):
 p=O/n;p.write_text(json.dumps(plain(v),indent=2,allow_nan=False)+'\n');p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return p
def multiply(A,B):return [[math.fsum(A[i][k]*B[k][j]for k in range(len(B)))for j in range(len(B[0]))]for i in range(len(A))]
def identity():return [[float(i==j)for j in range(4)]for i in range(4)]
def quat(q):
 x,y,z,w=q;norm=math.sqrt(math.fsum(v*v for v in q));x,y,z,w=[v/norm for v in q]
 return [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
def rotaxis(axis,angle):
 n=math.sqrt(math.fsum(v*v for v in axis));a=[v/n for v in axis];c=math.cos(angle);s=math.sin(angle);cross=[[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]]
 return [[c*(i==j)+(1-c)*a[i]*a[j]+s*cross[i][j]for j in range(3)]for i in range(3)]
def tf(R,p):
 T=identity()
 for i in range(3):T[i][:3]=R[i];T[i][3]=p[i]
 return T
def uv(c,K):return [K[0,0]*c[0]/c[2]+K[0,2],K[1,1]*c[1]/c[2]+K[1,2]]
M=read(F/'manifest.json');U=F/'robot/urdfs/g1/dex3_right.urdf';root=ET.parse(U).getroot();joints=list(root.findall('joint'));child={j.find('child').attrib['link']:j for j in joints};paths=read(P/'actual-URDF-branch-independence.json');rootlink=paths['root_link']
with np.load(F/'targets.npz')as z:q=z['robot_qpos']
with np.load(S/'e037-strict-source-wrist-mode-v1/UNQUALIFIED-strict-source-root-proposal.npz')as z:w=z['robot_qpos']
with np.load(F/'aligned-source-reference.npz')as z:cameras=z['camera_c2w'];human=z['hand_joints_world']
with np.load(F/'source-confidence.npz')as z:valid=z['hand_valid']
with np.load(S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz')as z:K=z['intrinsics']
stage=read(F/'command-proposal-report.json')['stage_rotation_from_inferred_support_normal'];caps={r['frame']:r for r in read(F.parent/'report.json')['source_rows']if r['hand']=='right'};records=read(P/'all-new-preserved-branch-source-returns.json');output=[];worlderrs=[];cameraerrs=[];pixelerrs=[]
with(O/'normalized-BEFORE-AFTER-journal.jsonl').open('w')as log:
 journal(log,dict(kind='BEFORE',operation='independent_inputs',producer_report_sha256=sha(P/'report.json'),saved_return_sha256=sha(P/'all-new-preserved-branch-source-returns.json'),actual_URDF_sha256=sha(U),no_shared_FK_or_producer_or_native_replayed=True,original_image_tolerance_px=1e-6))
 for rec in records['rows']+records['missing']:
  f=rec['frame'];link=rec['tip'];journal(log,dict(kind='BEFORE',operation='independent_chain_source_gate',frame=f,link=link,saved_return=rec));chain=[];cur=link
  while cur!=rootlink:
   j=child[cur];chain.append(j);cur=j.find('parent').attrib['link']
  chain.reverse();assert[j.attrib['name']for j in chain]==paths['paths'][link]
  assert not(set(paths['index_middle_joint_names'])&{j.attrib['name']for j in chain})
  T=tf(quat(w[f,1,3:7]),list(w[f,1,:3]));positions=dict(zip(M['joint_names']['right'],q[f,1,7:],strict=True));intermediate=[]
  for j in chain:
   origin=j.find('origin');xyz=[0.,0.,0.]if origin is None else list(map(float,origin.attrib.get('xyz','0 0 0').split()));rpy=[0.,0.,0.]if origin is None else list(map(float,origin.attrib.get('rpy','0 0 0').split()))
   Rx=rotaxis([1,0,0],rpy[0]);Ry=rotaxis([0,1,0],rpy[1]);Rz=rotaxis([0,0,1],rpy[2]);T=multiply(T,tf(multiply(multiply(Rz,Ry),Rx),xyz))
   if j.attrib['type']!='fixed':T=multiply(T,tf(rotaxis(list(map(float,j.find('axis').attrib['xyz'].split())),float(positions[j.attrib['name']])),[0.,0.,0.]))
   intermediate.append(dict(joint=j.attrib['name'],actual_transform=T));journal(log,dict(kind='AFTER',operation='independent_joint_transform',frame=f,link=link,returned=intermediate[-1]))
  p=[T[i][3]for i in range(3)];cam=cameras[f];stagep=[math.fsum(p[k]*stage[k][i]for k in range(3))for i in range(3)];rc=[math.fsum((stagep[k]-cam[k,3])*cam[k,i]for k in range(3))for i in range(3)];hc=[math.fsum((human[f,1,rec['landmark'],k]-cam[k,3])*cam[k,i]for k in range(3))for i in range(3)]
  we=max(abs(p[i]-rec['actual_URDF_world_point'][i])for i in range(3));ce=max(abs(rc[i]-rec['actual_camera_point'][i])for i in range(3));worlderrs.append(we);cameraerrs.append(ce);assert we<1e-12 and ce<1e-12
  available=bool(valid[f,1]and all(math.isfinite(x)for x in hc)and hc[2]>.01);assert available==rec['available']
  if not available:status='MISSING_preserved_no_gate_invented';error=None;excess=None
  elif not(all(math.isfinite(x)for x in rc)and rc[2]>.01):status='REJECTED_original_depth';error=None;excess=None
  else:
   ru=uv(rc,K);hu=uv(hc,K);error=math.hypot(*(ru[i]-hu[i]for i in range(2)));idx=0 if link=='right_hand_palm_link'else 1;cap=caps[f]['source_before_after_px'][0][idx];excess=error-cap;status='PASS_original_image'if error<=cap+1e-6 else'REJECTED_original_image';pe=abs(error-rec['error_px']);pixelerrs.append(pe);assert pe<1e-8
  assert status==rec['status'];cert=dict(frame=f,link=link,world_closure_m=we,camera_closure_m=ce,independent_error_px=error,independent_excess_px=excess,original_status=status,chain_independence_verified=True);save(f'certificate-{f:03d}-{link}.json',dict(chain_transforms=intermediate,certificate=cert));output.append(cert);journal(log,dict(kind='AFTER',operation='independent_chain_source_gate',returned=cert))
 summary=dict(schema='v2d.new-preserved-branch-source-independent.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_NEW_actual_URDF_projection_branch_and_original_gate_checks_passed=True,new_anchor_certificates=len(output),independent_actual_world_closure_m=max(worlderrs),independent_camera_closure_m=max(cameraerrs),independent_source_error_closure_px=max(pixelerrs),rejected_original_source_checks=sum(x['original_status'].startswith('REJECTED')for x in output),fixed_root_F_palm_thumb_combination_REJECTED=True,does_NOT_exclude_adjustable_thumb_allowed_root_or_original_family=True,all_raw_old_components_and_source_authority_UNCHANGED=True,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,wall_seconds=time.monotonic()-start)
 journal(log,dict(kind='AFTER',operation='independent_summary',returned=summary));summary['all_outputs_sha256']={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log','report.json']};save('report.json',summary);print(json.dumps(summary))
