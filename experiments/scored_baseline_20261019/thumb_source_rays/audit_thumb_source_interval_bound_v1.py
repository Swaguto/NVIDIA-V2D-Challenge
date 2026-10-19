"""NEW conservative fixed-root thumb/source/palm bound; no local solve or new reference."""
from thumb_ray_context_v1 import *
import datetime,heapq,math
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from v2d_dex3_flashchord.native_mesh import compiled_mesh_vertices
start=time.monotonic();O=S/'e037-thumb-source-interval-bound-v1';frames=[111,113];radius_gate=.0005;numeric_m=2e-12

def sin_interval(lo,hi,cosine=False):
 shift=math.pi/2 if cosine else 0.;lo+=shift;hi+=shift
 points=[lo,hi]+[math.pi/2+k*math.pi for k in range(math.ceil((lo-math.pi/2)/math.pi),math.floor((hi-math.pi/2)/math.pi)+1)]
 vals=[math.sin(p) for p in points];return np.array([min(vals)-3e-15,max(vals)+3e-15])
def product(a,b):
 values=np.array([a[0]*b[0],a[0]*b[1],a[1]*b[0],a[1]*b[1]]);return np.array([values.min()-numeric_m,values.max()+numeric_m])
def signed_scale(c,v):return np.array([min(c*v[0],c*v[1])-numeric_m,max(c*v[0],c*v[1])+numeric_m])
def linear_box(matrix,box,offset):
 lo=offset+np.sum(np.where(matrix>=0,matrix*box[:,0],matrix*box[:,1]),axis=1);hi=offset+np.sum(np.where(matrix>=0,matrix*box[:,1],matrix*box[:,0]),axis=1);return np.c_[lo-numeric_m,hi+numeric_m]
def source_pixel_box(frame,lower,upper):
 a,b,c=lower;aa,bb,cc=upper;sb=sin_interval(b,bb);cb=sin_interval(b,bb,True);sbc=sin_interval(b+c,bb+cc);cbc=sin_interval(b+c,bb+cc,True)
 x=np.array([-.0025,-.0025])+signed_scale(-.0458,sb)+signed_scale(-.035,sbc);y=np.array([.0193,.0193])+signed_scale(.0458,cb)+signed_scale(.035,cbc);rotated=np.stack([product(sin_interval(a,aa,True),x),y,signed_scale(-1.,product(sin_interval(a,aa),x))]);rotated[0]+=.0255
 camera=cameras[frame];A=camera[:3,:3].T@stage.T;rootR=Rotation.from_quat(q[frame,1,3:7]).as_matrix();offset=A@q[frame,1,:3]-camera[:3,:3].T@camera[:3,3];C=linear_box(A@rootR,rotated,offset)
 if C[2,0]<=.01:return np.array([[-np.inf,np.inf],[-np.inf,np.inf]])
 out=[]
 for axis in [0,1]:
  ratios=[C[axis,i]/C[2,j] for i in [0,1] for j in [0,1]];out.append([K[axis,axis]*min(ratios)+K[axis,2]-1e-7,K[axis,axis]*max(ratios)+K[axis,2]+1e-7])
 return np.array(out)
def image_lower_bound(pixel_box,observed_uv):
 distance=np.maximum(0.,np.maximum(pixel_box[:,0]-observed_uv,observed_uv-pixel_box[:,1]));return float(np.linalg.norm(distance))
def ry(a):return np.array([[math.cos(a),0,math.sin(a)],[0,1,0],[-math.sin(a),0,math.cos(a)]])
def rz(a):return np.array([[math.cos(a),-math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1]])
def sphere_lp(A,B):
 planes=np.r_[A,B];solution=linprog([0,0,0,-1],A_ub=np.c_[planes[:,:3],np.ones(len(planes))],b_ub=-planes[:,3],bounds=[(None,None)]*3+[(0,None)],method='highs')
 if not solution.success:return None
 p=solution.x[:3];r=float(np.min(-planes[:,:3]@p-planes[:,3]))-numeric_m
 if r<0:return None
 return p,r
# Independent analytical controls on intersection balls; these do not run old native jobs.
unit=np.r_[np.c_[np.eye(3),-np.ones(3)],np.c_[-np.eye(3),-np.ones(3)]];shift=unit.copy();shift[:,3]-=shift[:,0]*.5
control=sphere_lp(unit,shift);assert control is not None and abs(control[1]-.75)<1e-10
separate=unit.copy();separate[:,3]-=separate[:,0]*3;assert sphere_lp(unit,separate) is None
assert np.allclose(sin_interval(-math.pi/2,math.pi/2),[-1,1]);assert np.allclose(sin_interval(-.1,.1,True),[math.cos(.1),1]);controls=['overlapping cubes common ball .75m','disjoint cubes no common ball','sine interior extrema','cosine interior extrema']
# Bind native compiled support geometry into each body frame at the CURRENT state.
set_state(113);palm=canonical_id(bn,'right_hand_palm_link');b0=canonical_id(bn,'right_hand_thumb_0_link');b1=canonical_id(bn,'right_hand_thumb_1_link');assert model.body_parentid[b0]==palm and model.body_parentid[b1]==b0
for b in [b0,b1]:np.testing.assert_allclose(model.body_quat[b],[1,0,0,0],atol=1e-14)
p0=model.body_pos[b0].copy();p1=model.body_pos[b1].copy();planes=[]
for bid in [palm,b1]:
 gids=np.flatnonzero(model.geom_bodyid==bid);assert len(gids)==1;g=int(gids[0]);vs=compiled_mesh_vertices(model,int(model.geom_dataid[g]),mujoco_version=mujoco.__version__);world=vs@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g];local=(world-data.xpos[bid])@data.xmat[bid].reshape(3,3);h=ConvexHull(local);H=h.equations.copy();np.testing.assert_allclose(np.linalg.norm(H[:,:3],axis=1),1,atol=1e-12);planes.append(H)
PA,PB=planes;cache={}
def common_ball(lower,upper):
 center=(lower+upper)/2;a,b=center[:2];key=(float(a),float(b))
 if key not in cache:
  RR=ry(a)@rz(b);origin=p0+ry(a)@p1;NB=PB[:,:3]@RR.T;QB=np.c_[NB,PB[:,3]-NB@origin];cache[key]=sphere_lp(PA,QB)
 record=cache[key]
 if record is None:return -np.inf,None
 p,_=record;RR=ry(a)@rz(b);origin=p0+ry(a)@p1;local=RR.T@(p-origin);ra=float(np.min(-PA[:,:3]@p-PA[:,3]));rb=float(np.min(-PB[:,:3]@local-PB[:,3]));ha,hb=(upper-lower)[:2]/2;motion=2*np.linalg.norm(p-p0)*math.sin(ha/2)+2*np.linalg.norm(ry(a).T@(p-p0)-p1)*math.sin(hb/2);guaranteed=min(ra,rb-motion)-numeric_m
 return float(guaranteed),dict(center_angles=center.tolist(),common_point_palm_m=p.tolist(),center_face_margins_m=[ra,rb],maximum_thumb1_point_change_bound_m=float(motion),guaranteed_common_ball_radius_m=float(guaranteed))
rows=[];native_controls=[]
for frame in frames:
 lo,hi=bounds(frame);lower,upper=lo[4:],hi[4:];before=evaluate(frame,q[frame,1,7:]);assert before['source_mask'].all();camera=cameras[frame];obs=uv(((human[frame,1,4]-camera[:3,3])@camera[:3,:3])[None])[0];cap=float(before['errors'][1]);heap=[(0,0,lower.copy(),upper.copy())];counter=1;processed=0;source_rejected=0;collision_rejected=0;unresolved=[];witnesses=[];maxqueue=0
 while heap and processed<16000 and time.monotonic()-start<205:
  _,_,l,u=heapq.heappop(heap);processed+=1;maxqueue=max(maxqueue,len(heap));pixel_box=source_pixel_box(frame,l,u);lb=image_lower_bound(pixel_box,obs)
  if lb>cap+1e-6:source_rejected+=1;continue
  radius,witness=common_ball(l,u)
  if radius>radius_gate+1e-9:
   collision_rejected+=1
   if len(witnesses)<8:witnesses.append(witness)
   continue
  if np.max(u-l)<1e-4:unresolved.append(dict(lower=l.tolist(),upper=u.tolist(),image_error_lower_bound_px=lb,common_ball_radius_m=radius if np.isfinite(radius) else None));continue
  axis=int(np.argmax(u-l));mid=(l[axis]+u[axis])/2
  for which in [0,1]:
   ll=l.copy();uu=u.copy()
   if which==0:uu[axis]=mid
   else:ll[axis]=mid
   # Depth-first geometric subdivision; no candidate optimization or source-label tuning.
   heapq.heappush(heap,(-float(np.max(uu-ll)),counter,ll,uu));counter+=1
 complete=not heap;certified=bool(complete and not unresolved);record=dict(frame=frame,source_error_cap_px=cap,original_joint_box_rad=np.c_[lower,upper].tolist(),processed_boxes=processed,source_rejected_boxes=source_rejected,palm_collision_rejected_boxes=collision_rejected,unresolved_leaf_boxes=len(unresolved),queued_boxes_at_cap=len(heap),maximum_queue=maxqueue,enumeration_complete=complete,no_source_and_palm1mm_solution_in_declared_box=certified,unresolved_leaf_examples=unresolved[:6],common_ball_witness_examples=witnesses,finite_budget_or_unresolved_not_global_infeasibility=not certified);rows.append(record)
 # New native closure controls at one NEW interior angle for each study, no physics.
 middle=(lower+upper)/2;angles=q[frame,1,7:].copy();angles[4:]=middle;e=evaluate(frame,angles);R=ry(middle[0])@rz(middle[1]);o=p0+ry(middle[0])@p1;NB=PB[:,:3]@R.T;QB=np.c_[NB,PB[:,3]-NB@o];ball=sphere_lp(PA,QB)
 if ball is not None:
  depth=max([-c['signed_gap_m'] for c in e['self_contacts'] if set(c['native_bodies'])=={bn[palm],bn[b1]}],default=0.);native_controls.append(dict(frame=frame,angles_rad=middle.tolist(),common_ball_translation_lower_bound_m=2*ball[1],native_pair_penetration_m=depth,bound_respected=bool(2*ball[1]<=depth+1e-8)));assert native_controls[-1]['bound_respected']
 print(json.dumps(record),flush=True)
report=dict(schema='v2d.thumb-source-image-common-ball-interval-bound.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),context_sha256=sha(S/'compute/thumb_ray_context_v1.py'),native_mesh_reader_sha256=sha(R/'scripts/dex3_training/v2d_dex3_flashchord/native_mesh.py'),input_manifest_sha256=sha(F/'manifest.json'),native_model_sha256=sha(MP),frames=rows,analytical_controls_passed=controls,new_native_common_ball_controls=native_controls,scope='Necessary fixed-root source-image disk versus palm/thumb1 1mm geometry only; conservative trig/linear/fraction intervals with explicit outward numeric pads, and common inscribed sphere for actual compiled bodies. Missing boxes or budgets imply unresolved, not infeasibility; no motor/dynamics/contact-truth claim.',roundoff_padding_m=numeric_m,pixel_roundoff_padding_px=1e-7,all_source_human_authority_roots_model_commands_unchanged=True,no_completed_source_native_audit_or_optimization_repeated=True,no_native_compile_or_physics_steps=True,new_GPU_jobs=0,new_PPO_steps=0,whole_reference_or_physical_task_qualified=False,wall_seconds=time.monotonic()-start);(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(wall_seconds=report['wall_seconds'],summary=[{k:r[k] for k in ['frame','processed_boxes','enumeration_complete','unresolved_leaf_boxes','queued_boxes_at_cap','no_source_and_palm1mm_solution_in_declared_box']} for r in rows]),indent=2))
