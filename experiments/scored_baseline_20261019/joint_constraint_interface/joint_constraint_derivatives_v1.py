"""NEW complete directional/adjoint extension for the joint executable interface.

Uses completed NEW primitive contracts by hash, never replays their controls.
All tied joint AND continuous-direction slopes remain correlated by vertex id.
"""
from joint_constraint_interface_v1 import *
import hashlib


def full_support_return(A,B,nlocal,RA,RA_derivatives,phase,r):
    out=support_return(A,B,nlocal,RA,RA_derivatives,phase,r)
    if out['status']!='RETURNED':return out
    ia,ib=out['A_ties'],out['B_ties']
    out['ALL_tie_direction_derivatives']=dict(A=A[0][ia]@RA,B=-B[0][ib]@RA)
    return out


def support_directional(out,command_direction,local_direction,phase_direction,r):
    if out['status']!='RETURNED':return dict(status='UNKNOWN_no_support_return')
    ad=out['ALL_tie_direction_derivatives']['A']@np.asarray(local_direction)
    bd=out['ALL_tie_direction_derivatives']['B']@np.asarray(local_direction)
    for col,slopes in out['ALL_tie_factor_derivatives'].items():
        v=command_direction.get(int(col),0.)
        ad=ad+slopes['A']*v;bd=bd+slopes['B']*v
    return dict(status='RETURNED',ALL_A_tied_slopes=ad,ALL_B_tied_slopes=bd,
        directional_residual=float(r*phase_direction-np.max(ad)-np.max(bd)),
        directional_unit_equality=float(out['unit_J']@local_direction),
        SEARCH_mean_selection_NOT_directional_derivative=True)


def cone_directional(block,dblock):
    tail=np.asarray(block)[1:];dtail=np.asarray(dblock)[1:];norm=np.linalg.norm(tail)
    return float(dblock[0]-(tail@dtail/norm if norm else np.linalg.norm(dtail)))


class JointExecutableInterface(JointInterface):
    def linear_actions(self,v):
        v=np.asarray(v)
        return dict(object_slack_direction=-self.A@v[465:],
                    robot_linear_slack_direction=-self.robot@v[:465],objective_Hv=self.P@v)

    def linear_adjoints(self,object_dual,robot_dual):
        out=np.zeros(1523)
        out[465:]=-self.A.T@np.asarray(object_dual)
        out[:465]=-self.robot.T@np.asarray(robot_dual)
        return out

    def evaluate_NEW(self,x,directions,proposal_path,record):
        if not isinstance(record,DurableRecorder):raise ValueError('durable fsync recorder required')
        x=np.asarray(x);directions=np.asarray(directions)
        proposal=json.loads(Path(proposal_path).read_text())
        if x.shape!=(1523,) or directions.shape!=(151,1653,3):raise ValueError('complete arrays required')
        if not(np.isfinite(x).all() and np.isfinite(directions).all()):raise ValueError('nonfinite UNKNOWN')
        required=dict(x_bytes_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
            directions_bytes_sha256=hashlib.sha256(directions.tobytes()).hexdigest(),
            interface_manifest_sha256=sha(self.root/'NEW-interface-input-bindings-RETURN.json'),
            executable_derivative_sha256=sha(C/'joint_constraint_derivatives_v1.py'))
        if any(proposal.get(k)!=v for k,v in required.items()):raise ValueError('NEW full proposal/provenance not sealed')
        record('NEW-input-provenance-BEFORE',dict(x=x,directions=directions,proposal=proposal))
        canonical=self.canonical(x);record('NEW-ALL-canonical-RETURN',canonical)
        source=self.source_and_domains(x);record('NEW-ALL-source-and-halfdomains-RETURN',source)
        motion=self.actual_object_motion(x);record('NEW-ALL6-object-motion-RETURN',motion)
        for frame in range(151):
            body=self.geometry_frame(x,frame);record(f'NEW-frame-{frame:03d}-body-RETURN',body)
            pairs=[]
            for p,(a,b) in enumerate(self.incidence['geom_pair']):
                a,b=int(a),int(b);ba=self.raw['geom_bodyid'][a]
                support=full_support_return(body['geometry'][a],body['geometry'][b],directions[frame,p],body['rotations'][ba],body['rotation_J'][a],x[1522],self.r)
                pairs.append(dict(pair=p,conditional_filter=self.flags[p]['conditional_builtin_filter_reasons'],ALL_support_RETURN=support))
            record(f'NEW-frame-{frame:03d}-ALL-pairs-RETURN',pairs)
        return dict(status='RETURNS_ONLY_NOT_QUALIFIED',ALLpairscopes=249603,
          ALLties_ANDcontinuous_direction_slopes_retained=True,
          actual_integercell_strictdepth_training_NN_rank_median_quality_and_original_fullpixel_heldout_POST_STILL_REQUIRED=True,
          original_whole_URDF_robot_root_finger_motion_native_self_swept_motor_grasp_STILL_REQUIRED=True)
