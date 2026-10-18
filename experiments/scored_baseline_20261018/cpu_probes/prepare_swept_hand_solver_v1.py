"""Prepare a new bounded experiment; original executed source remains intact."""
from pathlib import Path
C=Path(__file__).resolve().parent
p=(C/'plane_aligned_articulated_trajectory_v2.py').read_text()
p=p.replace("OUT=S/'plane-aligned-articulated-trajectory-v2'","OUT=S/'plane-aligned-articulated-trajectory-v3'")
p=p.replace("v2d.plane-aligned-articulated-source-reference.v2","v2d.plane-aligned-articulated-source-reference.v3")
p=p.replace("with np.load(P/'targets.npz',allow_pickle=False) as z:retained_commands=z['robot_qpos'].copy()","with np.load(S/'plane-aligned-articulated-trajectory-v2/inferred-robot-command-candidate.npz',allow_pickle=False) as z:retained_commands=z['robot_qpos'].copy()")
p=p.replace("warm,_=measure(x,add_planes=True)","warm,_=measure(x,add_planes=True)")
insert='''
# Separate between-knot diagnostic constraints. No extra source observations,
# no change to the source/native/motion gates or absolute morph bounds.
sweep_specs=[]
for frame,(a,b) in enumerate(zip(original['timestamps_s'][:-1],original['timestamps_s'][1:])):
    steps=int(np.ceil((b-a)/.002))
    sweep_specs.extend((frame,j/steps) for j in range(1,steps))
sweep_planes={}
quaternion_addresses=[]
for j in range(model.njnt):
    if model.jnt_type[j]==mujoco.mjtJoint.mjJNT_BALL:quaternion_addresses.append(int(model.jnt_qposadr[j]))
    elif model.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE:quaternion_addresses.append(int(model.jnt_qposadr[j])+3)
def interpolated_qpos(frame,alpha,all_x):
    first=native_qpos(frame,all_x[frame]);second=native_qpos(frame+1,all_x[frame+1])
    result=(1-alpha)*first+alpha*second
    for adr in quaternion_addresses:
        r=Rotation.from_quat(np.stack([first[adr:adr+4],second[adr:adr+4]])[:,[1,2,3,0]])
        relative=(r[0].inv()*r[1]).as_rotvec()
        result[adr:adr+4]=(r[0]*Rotation.from_rotvec(alpha*relative)).as_quat()[[3,0,1,2]]
    return result
def set_sweep(frame,alpha,all_x):
    data.qpos[:]=interpolated_qpos(frame,alpha,all_x);mujoco.mj_fwdPosition(model,data)
def swept_tangent(frame,alpha,all_x,si):
    result=np.zeros((model.nv,26))
    for endpoint in range(2):
        for j in range(13):
            minus=all_x.copy();plus=all_x.copy()
            minus[frame+endpoint,si,j]-=1e-5;plus[frame+endpoint,si,j]+=1e-5
            a=interpolated_qpos(frame,alpha,minus);b=interpolated_qpos(frame,alpha,plus)
            velocity=np.zeros(model.nv);mujoco.mj_differentiatePos(model,velocity,2e-5,a,b)
            result[:,endpoint*13+j]=velocity
    assert np.isfinite(result).all()
    return result
def measure_sweep(all_x,add_planes=False):
    failed=0;maximum=0.;minimum=.1;added=0
    for idx,(frame,alpha) in enumerate(sweep_specs):
        deadline();set_sweep(frame,alpha,all_x);depth=0.
        for c in data.contact[:data.ncon]:
            a,b=int(c.geom1),int(c.geom2)
            for side,h in binding.items():
                if a in h['hand_geoms'] and b in target_geoms:hand,target,normal=a,b,-np.array(c.frame[:3])
                elif b in h['hand_geoms'] and a in target_geoms:hand,target,normal=b,a,np.array(c.frame[:3])
                else:continue
                depth=max(depth,-float(c.dist))
                if add_planes:
                    entries=sweep_planes.setdefault((idx,h['si']),[])
                    if not any(p['hand']==hand and p['target']==target for p in entries):
                        support=float(np.einsum('ni,i->n',vertices(target),normal,optimize=False).max())
                        entries.append(dict(hand=hand,target=target,normal=normal,support=support,active_vertex_indices=set()));added+=1
        maximum=max(maximum,depth);failed+=depth>.001
        for si in range(2):
            for p in sweep_planes.get((idx,si),[]):
                gaps=np.einsum('ni,i->n',vertices(p['hand']),p['normal'],optimize=False)-p['support']
                minimum=min(minimum,float(gaps.min()))
                if add_planes and (not p['active_vertex_indices'] or gaps.min()<.0005-1e-7):p['active_vertex_indices'].update(np.flatnonzero(gaps<=gaps.min()+1e-9).tolist())
    return dict(interior_samples=len(sweep_specs),samples_over_1mm=int(failed),maximum_native_penetration_m=maximum,minimum_conservative_plane_gap_m=minimum,new_planes=added)
report['sweep_control']=measure_sweep(x,add_planes=True)
report['sweep_method']='Linear root/finger and shortest-arc quaternion interpolation <=2ms; discrete diagnostic, no continuous-collision or physics certificate.'
report['between_knot_source_observations_fabricated']=False
save()
'''
p=p.replace("eye=sparse.eye(n,format='csc')",insert+"\neye=sparse.eye(n,format='csc')")
extra='''
        # Linearize conservative convex planes at each detected interior
        # intersection. Endpoints remain the only optimization variables.
        for (idx,si),entries in sweep_planes.items():
            frame,alpha=sweep_specs[idx];set_sweep(frame,alpha,x)
            T=swept_tangent(frame,alpha,x,si)
            current=np.r_[x[frame,si],x[frame+1,si]]
            column_indices=np.r_[np.arange(frame*26+si*13,frame*26+si*13+13),np.arange((frame+1)*26+si*13,(frame+1)*26+si*13+13)]
            for p in entries:
                geom=p['hand'];verts=vertices(geom)[sorted(p['active_vertex_indices'])];r=verts-data.geom_xpos[geom];normal=p['normal']
                jp=np.zeros((3,model.nv));jr=jp.copy();mujoco.mj_jac(model,data,jp,jr,data.geom_xpos[geom],int(model.geom_bodyid[geom]))
                gradient=mm(np.einsum('i,ij->j',normal,jp,optimize=False)[None,:]+mm(np.cross(r,normal),jr),T)/.01
                gap=(np.einsum('ni,i->n',verts,normal,optimize=False)-p['support'])/.01
                firstrow=len(low);rr,cc=np.nonzero(abs(gradient)>1e-14)
                rows.extend((rr+firstrow).tolist());cols.extend(column_indices[cc].tolist());values.extend(gradient[rr,cc].tolist())
                low.extend((.05-gap+mm(gradient,current)).tolist());high.extend([np.inf]*len(gradient))
'''
p=p.replace("        A=sparse.csc_matrix((values",extra+"\n        A=sparse.csc_matrix((values")
p=p.replace("entry.update(measurements);report", "swept=measure_sweep(x,add_planes=True);entry['swept']=swept\n        entry.update(measurements);report")
p=p.replace("if measurements['source_projection_passed']", "if swept['samples_over_1mm']==0 and swept['minimum_conservative_plane_gap_m']>=.0005-1e-7 and measurements['source_projection_passed']")
p=p.replace("    final,commands=measure(x)","    final,commands=measure(x)\n    report['sweep_final']=measure_sweep(x)")
p=p.replace("bool(final['source_projection_passed']", "bool(report['sweep_final']['samples_over_1mm']==0 and report['sweep_final']['minimum_conservative_plane_gap_m']>=.0005-1e-7 and final['source_projection_passed']")
p=p.replace('for iteration in range(12):','for iteration in range(15):')
(C/'plane_aligned_articulated_trajectory_v3.py').write_text(p)
g=(C/'guard_plane_aligned_articulated_trajectory_v2.py').read_text().replace('plane-aligned-articulated-trajectory-v2','plane-aligned-articulated-trajectory-v3').replace('plane_aligned_articulated_trajectory_v2.py','plane_aligned_articulated_trajectory_v3.py')
(C/'guard_plane_aligned_articulated_trajectory_v3.py').write_text(g)
import ast
ast.parse(p);ast.parse(g)
print('Prepared new sweep-constrained endpoint solver with unchanged bounds/gates')
