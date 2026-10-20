"""NEW executable coupled interface. Construction numerics are NOT gate proofs.

Actual episode calls require a new sealed proposal and a durable recorder.
Synthetic contracts use separate toy inputs. No optimizer is selected here.
"""
from pathlib import Path
from fractions import Fraction
import numpy as np
from scipy.spatial.transform import Rotation
from scipy import sparse
from joint_collision_maps_v1 import S,C,F,M,G,O,sha,save,save_npz,skew
from coupled_angle_source_v1 import camera_source_factor
from full_pose_construction_v1 import actual_motion
import json


def cayley_terms(eta_scaled,u,r,scale):
    t=np.asarray(u)/scale
    eta=np.asarray(eta_scaled)*r
    den=1+t@t
    num=(1-t@t)*np.eye(3)+2*np.outer(t,t)+2*skew(t)
    Q=num/den
    derivatives=[]
    for j in range(3):
        e=np.eye(3)[j]
        dn=-2*t[j]*np.eye(3)+2*(np.outer(e,t)+np.outer(t,e))+2*skew(e)
        derivatives.append((dn*den-num*(2*t[j]))/(den*den*scale))
    dt_eta=np.eye(3)*r/den
    dt_u=np.column_stack([-2*t[j]*eta/(den*den*scale) for j in range(3)])
    return Q,eta/den,np.asarray(derivatives),np.column_stack([dt_eta,dt_u])


def body_tree(raw,qvalues):
    """Native joint order; finite quaternion normalization follows native scope.

    This analytical real map is not installed-binary FK equivalence. Commands
    remain unchanged. Returned axes/pivots define real search derivatives.
    """
    n=raw['nbody'];R=np.tile(np.eye(3),(n,1,1));p=np.zeros((n,3))
    axes={};pivots={}
    quat=lambda q:Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix()
    for b in range(1,n):
        parent=raw['body_parentid'][b]
        Rb=R[parent]@quat(raw['body_quat'][b]);pb=p[parent]+R[parent]@raw['body_pos'][b]
        count=raw['body_jntnum'][b];start=raw['body_jntadr'][b]
        for j in range(start,start+count):
            kind=raw['jnt_type'][j];value=np.asarray(qvalues[j]);address=raw['jnt_qposadr'][j]
            if kind==0:
                pb=value[:3].copy();Rb=quat(value[3:]);continue
            axis=Rb@raw['jnt_axis'][j];pivot=pb+Rb@raw['jnt_pos'][j]
            axes[j]=axis;pivots[j]=pivot
            if kind==2:
                pb=pb+axis*(float(value)-raw['qpos0'][address])
            elif kind in [1,3]:
                if kind==1:local=quat(value)
                else:local=Rotation.from_rotvec(np.asarray(raw['jnt_axis'][j])*(float(value)-raw['qpos0'][address])).as_matrix()
                Rb=Rb@local;pb=pivot-Rb@raw['jnt_pos'][j]
            else:raise ValueError('unsupported native joint')
        R[b]=Rb;p[b]=pb
    return R,p,axes,pivots


def support_return(A,B,nlocal,RA,RA_derivatives,phase,r):
    """ALL vertices/ties, body-A-local continuous direction, full product rule.

    Vertex/rotation derivative dictionaries key GLOBAL command columns. The mean
    of all tied slopes is a SEARCH Clarke selection, not every directional slope.
    No direction normalization or candidate/slack/dual repair occurs.
    """
    nlocal=np.asarray(nlocal);n=RA@nlocal
    va,ja=A;vb,jb=B
    if n.shape!=(3,) or not np.isfinite(n).all() or np.linalg.norm(n)==0:
        return dict(status='UNKNOWN_zero_or_nonfinite_direction')
    sa=va@n;sb=vb@(-n);ia=np.flatnonzero(sa==np.max(sa));ib=np.flatnonzero(sb==np.max(sb))
    a=va[ia].mean(axis=0);b=vb[ib].mean(axis=0)
    cols=sorted(set(ja)|set(jb)|set(RA_derivatives))
    all_tie_derivatives={};mean={}
    for c in cols:
        da=ja[c][ia] if c in ja else np.zeros((len(ia),3))
        db=jb[c][ib] if c in jb else np.zeros((len(ib),3))
        dn=RA_derivatives[c]@nlocal if c in RA_derivatives else np.zeros(3)
        # Separate A and B tie terms factor the complete Cartesian tie set.
        aa=da@n+va[ia]@dn
        bb=db@(-n)+vb[ib]@(-dn)
        all_tie_derivatives[str(c)]=dict(A=aa,B=bb)
        mean[str(c)]=-float(aa.mean()+bb.mean())
    return dict(status='RETURNED',support_A=float(sa.max()),support_B=float(sb.max()),
        residual=float(.001+r*phase-sa.max()-sb.max()),
        unit_equality=float(nlocal@nlocal-1),unit_J=2*nlocal,
        n_world=n,n_local=nlocal,A_ties=ia,B_ties=ib,
        ALL_tie_factor_derivatives=all_tie_derivatives,
        SEARCH_command_residual_J=mean,
        SEARCH_direction_residual_J=-(RA.T@(a-b)),phase_residual_J=float(r),
        mean_tie_selection_NOT_universal_directional_derivative=True)


def source_chain(records,root,angles,link):
    children={j['child']:j for j in records};chain=[];cur=link
    while cur in children:
        j=children[cur];chain.append(j);cur=j['parent']
    T=np.eye(4);T[:3,:3]=Rotation.from_quat(root[3:7]).as_matrix();T[:3,3]=root[:3]
    axes={};pivots={}
    for rec in chain[::-1]:
        local=np.eye(4);local[:3,:3]=Rotation.from_euler('xyz',rec['rpy']).as_matrix();local[:3,3]=rec['xyz'];T=T@local
        if rec['type']!='fixed':
            axes[rec['name']]=T[:3,:3]@rec['axis'];pivots[rec['name']]=T[:3,3].copy()
            motion=np.eye(4);motion[:3,:3]=Rotation.from_rotvec(np.asarray(rec['axis'])*angles[rec['name']]).as_matrix();T=T@motion
    point=T[:3,3]
    jac={name:np.cross(a,point-pivots[name]) for name,a in axes.items()}
    return point,jac


class DurableRecorder:
    def __init__(self,directory):
        self.directory=Path(directory)
        if not self.directory.is_dir():raise ValueError('new guarded output directory required')
        self.count=0
    def __call__(self,name,value):
        path=self.directory/f'{self.count:06d}-{name}.json'
        save(path,value)
        self.count+=1


class JointInterface:
    """Inputs are immutable NEW routed artifacts, and old by-hash authorities."""
    def __init__(self,root):
        self.root=Path(root)
        self.manifest=json.loads((self.root/'NEW-interface-input-bindings-RETURN.json').read_text())
        for path,digest in self.manifest['input_sha256'].items():
            if sha(path)!=digest:raise ValueError('immutable input changed '+path)
        self.A=sparse.load_npz(self.root/'NEW-object-retained-A-RETURN.npz')
        self.robot=sparse.load_npz(self.root/'NEW-robot-linear-A-RETURN.npz')
        self.P=sparse.load_npz(self.root/'NEW-embedded-objective-P-RETURN.npz')
        with np.load(self.root/'NEW-interface-constants-RETURN.npz',allow_pickle=False) as z:
            self.constants={k:z[k] for k in z.files}
        self.raw=json.loads((O/'NEW-static-native-tree-RETURN.json').read_text())
        self.program=json.loads((O/'NEW-articulated-chain-program-RETURN.json').read_text())
        self.urdf=json.loads((O/'NEW-URDF-native-incidence-RETURN.json').read_text())['URDF']['right']
        self.variables=json.loads((M/'continuous-angle-variables.json').read_text())
        self.lookup={(v['frame'],v['URDF_joint']):v['column'] for v in self.variables}
        self.named={v['name']:v['column'] for v in self.variables}
        self.source=json.loads((M/'all-direct-source-factor-proposals.json').read_text())
        self.domains=json.loads((M/'retained-halfangle-domains-and-unknown-source-evidence.json').read_text())
        self.linear_exact=json.loads((M/'all-exact-linear-angle-inequalities.json').read_text())
        with np.load(O/'NEW-complete-column-and-pair-incidence-RETURN.npz',allow_pickle=False) as z:
            self.incidence={k:z[k] for k in z.files}
        with np.load(M/'unqualified-model-fixed-components-and-initial-F.npz',allow_pickle=False) as z:
            self.fixed=z['model_base_qpos'];self.times=z['timestamps_s']
        with np.load(G/'NEW-native-body-rotation-factors-RETURN.npz',allow_pickle=False) as z:
            self.native_base=z['cached_body_poses'][:,29];self.stage=z['stage_world_translation_linear']
            self.vertices={g:z[f'g{g}_vertices_body'] for g in range(58)}
            self.normals={g:z[f'g{g}_normals_body'] for g in range(58)}
        with np.load(G/'NEW-source-rotation-factors-RETURN.npz',allow_pickle=False) as z:
            self.world_base=z['original_world_from_object'];self.timestamps=z['original_timestamps_s']
        self.r=float(self.constants['radius']);self.scale=float(self.constants['scale'])
        self.flags=json.loads((S/'e037-joint-collision-predicates-v1/NEW-ALL-pair-quantified-geometric-constraints-RETURN.json').read_text())['records']

    def canonical(self,x):
        x=np.asarray(x)
        if x.shape!=(1523,) or not np.isfinite(x).all():raise ValueError('finite complete NEW coordinates required')
        obj=self.A@x[465:]
        slack=self.constants['object_b']-obj
        robot=self.constants['robot_b']-self.robot@x[:465]
        obj_J=lambda v:-self.A@v[465:]
        nn=int(self.constants['nonnegative']);cones=[];offset=nn
        for width in [4]*900+[5]*151:
            block=slack[offset:offset+width]
            cones.append(dict(start=offset,width=width,value=float(block[0]-np.linalg.norm(block[1:])),zero_tail=bool(np.linalg.norm(block[1:])==0)))
            offset+=width
        if offset!=len(slack):raise ValueError('ALL cone rows required')
        return dict(object_slack=slack,robot_linear_slack=robot,cones=cones,
            objective=float(.5*x@(self.P@x)+self.constants['q']@x),
            objective_J=self.P@x+self.constants['q'],all_object_rows=len(slack),all_robot_rows=len(robot))

    def geometry_frame(self,x,frame):
        qvalues={}
        for rec in self.program['q_assignments']:
            j=rec['joint'];kind=rec['kind']
            if kind=='hinge_angle':
                idx=rec['fixed_indices'];column=int(self.incidence['column_for_frame_joint'][frame,j])
                qvalues[j]=x[column] if column>=0 else self.fixed[frame,idx[1],idx[2]]
            elif kind=='fixed_saved_root_command':
                idx=rec['indices'];qvalues[j]=self.fixed[frame,idx[1],idx[2]]
                if len(idx[2])==1:qvalues[j]=float(qvalues[j][0])
            elif kind=='ORIGINAL_qpos0_support':
                a=rec['qpos0_address'];qvalues[j]=np.asarray(self.raw['qpos0'][a:a+7])
            elif kind=='object_Cayley_from_F':
                # Placeholder is never used: object rigid map is explicitly replaced
                # below; native object/support free roots have no robot descendants.
                a=rec['qpos0_address'];qvalues[j]=np.asarray(self.raw['qpos0'][a:a+7])
        rotations,positions,axes,pivots=body_tree(self.raw,qvalues)
        values=x[465+7*frame:465+7*frame+7]
        Q,delta,dQ,dt=cayley_terms(values[:3],values[3:6],self.r,self.scale)
        rotations[29]=self.native_base[frame,:3,:3]@Q
        positions[29]=self.native_base[frame,:3,3]+self.stage@delta
        transformed=[];rotation_J=[];normal_maps=[]
        for geom in range(58):
            b=self.raw['geom_bodyid'][geom];R=rotations[b];p=positions[b]
            points=self.vertices[geom]@R.T+p;J={};RJ={}
            if b==29:
                for k in range(6):
                    col=465+7*frame+k
                    rot=np.zeros((3,3)) if k<3 else self.native_base[frame,:3,:3]@dQ[k-3]
                    J[col]=self.vertices[geom]@rot.T+self.stage@dt[:,k]
                    RJ[col]=rot
            else:
                for j in range(self.raw['njnt']):
                    c=int(self.incidence['column_for_frame_joint'][frame,j])
                    if c>=0 and self.incidence['body_joint_incidence'][b,j]:
                        J[c]=np.cross(axes[j],points-pivots[j]);RJ[c]=skew(axes[j])@R
            transformed.append((points,J));rotation_J.append(RJ)
            normal_maps.append(self.normals[geom]@R.T)
        return dict(geometry=transformed,rotations=rotations,positions=positions,
                    rotation_J=rotation_J,ALL_rotated_body_normals=normal_maps)

    def source_and_domains(self,x):
        records=[]
        names=[j['name'] for j in self.urdf if j['type']!='fixed']
        manifest=json.loads((F/'manifest.json').read_text())
        ordered=manifest['joint_names']['right']
        for rec in self.source:
            angles=dict(zip(ordered,rec['fixed_F_full_finger_values'],strict=True))
            for name in rec['free_joint_names']:angles[name]=x[self.lookup[rec['frame'],name]]
            point,jac=source_chain(self.urdf,np.asarray(rec['root_pose']),angles,rec['link'])
            cols=[self.lookup[rec['frame'],name] for name in rec['free_joint_names']]
            matrix=np.column_stack([jac[name] for name in rec['free_joint_names']])
            out=camera_source_factor(point,matrix,rec['camera'],rec['stage'],rec['K'],rec['source_pixel'],rec['ORIGINAL_cap_px'],rec['ORIGINAL_image_tolerance_px'],rec['ORIGINAL_minimum_depth_m'])
            records.append(dict(factor=rec['id'],columns=cols,returned=out))
        bounds=[]
        for rec in self.domains['retained_domain_factors']:
            c=self.named[rec['angle_variable']];angle=float(x[c]);co=float(np.cos(angle/2))
            if co==0 or not np.isfinite(co):bounds.append(dict(column=c,status='UNKNOWN_pole'));continue
            value=float(np.tan(angle/2));lo,hi=[r['approx'] for r in rec['half_angle_bounds_COPIED']]
            bounds.append(dict(column=c,half_angle=value,lower_residual=value-lo,upper_residual=hi-value,
                              derivative=float(.5/(co*co)),finite_chart=True))
        return dict(ALL194source=records,ALL38half_domains=bounds,
            full_region2_and_3unknown109bands_RETAINED_BY_HASH=True,
            old_polynomials_charts_NOT_regenerated=True,unknown_bands_NOT_false_or_deleted=True,
            no_source_accuracy_or_whole_source_qualification=True)

    def actual_object_motion(self,x):
        poses=[]
        for frame in range(151):
            values=x[465+7*frame:465+7*frame+7]
            Q,delta,_,_=cayley_terms(values[:3],values[3:6],self.r,self.scale)
            T=self.world_base[frame].copy()
            T[:3,:3]=T[:3,:3]@Q;T[:3,3]+=delta
            poses.append(T)
        return actual_motion(np.asarray(poses),self.timestamps,self.constants['object_caps'])

    def evaluate_NEW(self,x,directions,proposal_path,record):
        """ONE new input only after declaration; not called by map preparation."""
        if not isinstance(record,DurableRecorder):raise ValueError('durable fsync recorder required')
        x=np.asarray(x);directions=np.asarray(directions)
        proposal=json.loads(Path(proposal_path).read_text())
        if x.shape!=(1523,) or directions.shape!=(151,1653,3):raise ValueError('complete joint/direction arrays required')
        if not(np.isfinite(x).all() and np.isfinite(directions).all()):raise ValueError('unknown nonfinite')
        if proposal['x_bytes_sha256']!=__import__('hashlib').sha256(x.tobytes()).hexdigest() or proposal['directions_bytes_sha256']!=__import__('hashlib').sha256(directions.tobytes()).hexdigest():raise ValueError('unsealed NEW input')
        record('NEW-input-BEFORE',dict(x=x,directions=directions,proposal=proposal))
        canonical=self.canonical(x);record('NEW-ALL-canonical-RETURN',canonical)
        source=self.source_and_domains(x);record('NEW-ALL-robot-source-domain-RETURN',source)
        motion=self.actual_object_motion(x);record('NEW-ALL6-actual-object-motion-RETURN',motion)
        for frame in range(151):
            body=self.geometry_frame(x,frame);record(f'NEW-frame-{frame:03d}-body-RETURN',body)
            pairs=[]
            for p,(a,b) in enumerate(self.incidence['geom_pair']):
                a,b=int(a),int(b);ba=self.raw['geom_bodyid'][a]
                support=support_return(body['geometry'][a],body['geometry'][b],directions[frame,p],body['rotations'][ba],body['rotation_J'][a],x[1522],self.r)
                pairs.append(dict(pair=p,conditional_filter=self.flags[p]['conditional_builtin_filter_reasons'],all_support_RETURN=support))
            record(f'NEW-frame-{frame:03d}-ALL-pairs-RETURN',pairs)
        return dict(status='RETURNS_ONLY_NOT_QUALIFIED',canonical=canonical,source=source,actual_object_motion=motion,
                    original_actual_training_fullpixel_heldout_motion_native_self_swept_motor_grasp_STILL_REQUIRED=True)
