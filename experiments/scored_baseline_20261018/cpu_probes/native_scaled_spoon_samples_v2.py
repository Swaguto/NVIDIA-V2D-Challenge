"""Fresh native import and seven-knot audit of the learned-depth scale arm.

The original collision decomposition remains byte-identical. Uniform URDF mesh
scale applies to those hulls through the unchanged upstream importer; a separate
effective scaled OBJ supplies the matching evaluation surface. No physics step,
full-trajectory gate or training qualification is claimed.
"""
import hashlib
import json
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path[:0]=[str(S/'cpu-native-runtime-v1/src'),str(R/'scripts/dex3_training'),str(S/'cpu-native-extra-deps-v1')]
from check_native_scene import compile_scene
from audit_hand_reference_contacts import canonical_id
from v2d_dex3_flashchord.kinematics import URDFKinematics

P=S/'articulated-source-contact-native-validation-v1/frozen'
original=S/'scored-cohort-14-results/policy-e037/frozen'
fit_path=S/'fast-stereo-scaled-seven-pose-probe-v1/report.json'
fit=json.loads(fit_path.read_text());assert len(fit['rows'])==7
assert sorted(row['frame'] for row in fit['rows'])==[0,25,50,75,100,125,150]
OUT=S/'native-scaled-spoon-samples-v2';OUT.mkdir(exist_ok=True);inputs=OUT/'unbound'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((P/'manifest.json').read_text())
for name,h in manifest['files'].items():assert sha(P/name)==h
shutil.copytree(P,inputs)
scale=1.3
urdf=inputs/'objects/wooden_spoon.urdf';tree=ET.parse(urdf)
for node in tree.getroot().iter('mesh'):
    values=np.fromstring(node.attrib.get('scale','1 1 1'),sep=' ')*scale
    node.set('scale',' '.join(format(v,'.17g') for v in values))
for inertial in tree.getroot().iter('inertial'):
    origin=inertial.find('origin')
    if origin is not None:origin.set('xyz',' '.join(format(v,'.17g') for v in np.fromstring(origin.attrib.get('xyz','0 0 0'),sep=' ')*scale))
    inertia=inertial.find('inertia')
    for key,value in inertia.attrib.items():inertia.set(key,format(float(value)*scale**2,'.17g'))
tree.write(urdf,encoding='utf-8',xml_declaration=True)
source_mesh=trimesh.load(inputs/'objects/wooden_spoon.obj',force='mesh',process=False)
effective=source_mesh.copy();effective.vertices=np.asarray(source_mesh.vertices)*scale
eval_path=inputs/'objects/wooden_spoon-eval-scaled.obj';effective.export(eval_path)
manifest['objects'][0]['mesh']='objects/wooden_spoon-eval-scaled.obj'
manifest['objects'][0]['radius_m']*=scale
proposal=dict(schema='v2d.uniform-own-geometry-scale-proposal.v1',uniform_scale=scale,source_mesh_sha256=sha(P/'objects/wooden_spoon.obj'),unscaled_collision_source='objects/wooden_spoon.obj',effective_evaluation_mesh='objects/wooden_spoon-eval-scaled.obj',urdf_uniform_mesh_scale=True,unchanged_decomposition_cache=False,new_decomposition_produced_by_unchanged_upstream=True,nominal_mass_changed=False,nominal_inertia_scaled_by=scale**2,mass_and_inertia_independently_measured=False,source_fits_report_sha256=sha(fit_path),full_trajectory_qualified=False,physical_task_accepted=False)
(inputs/'geometry-scale-proposal.json').write_text(json.dumps(proposal,indent=2)+'\n')
manifest['geometry_scale_proposal']='geometry-scale-proposal.json'
manifest['prior_hand_contact_validation_recheck_required_after_geometry_change']=True
manifest['files']={p.relative_to(inputs).as_posix():sha(p) for p in sorted(inputs.rglob('*')) if p.is_file() and p.name!='manifest.json'}
(inputs/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
# The importer bakes scale into vertices before cache lookup. Generate a new
# cache with the original upstream producer; keep the failed v1 and old cache.
shutil.rmtree(inputs/'collision-cache')
(inputs/'collision-cache-receipt.json').unlink()
manifest.pop('collision_cache')
manifest['files']={p.relative_to(inputs).as_posix():sha(p) for p in sorted(inputs.rglob('*')) if p.is_file() and p.name!='manifest.json'}
(inputs/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
from freeze_collision_cache import freeze
cache_result=freeze(inputs,OUT/'frozen',seed_cache=P/'collision-cache')
inputs=OUT/'frozen';manifest=json.loads((inputs/'manifest.json').read_text())
report=dict(schema='v2d.native-scaled-spoon-sample-audit.v2',producer_sha256=sha(Path(__file__)),input_manifest_sha256=sha(inputs/'manifest.json'),source_fits_report_sha256=sha(fit_path),uniform_scale=scale,physics_stepped=False,training_steps=0,full_reference_contact_gate_completed=False,trajectory_promoted=False,physical_task_accepted=False,source_confidence_modified=False,original_source_files_modified=False,frames=[])
save=lambda:(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');started=time.monotonic();save()
try:
    compiled=compile_scene(inputs);model=compiled[2].mj_model
    mujoco.mj_saveModel(model,str(OUT/'native-model.mjb'),None)
    control=mujoco.MjModel.from_binary_path(str(S/'local-native-contact-control-v1/native-model.mjb'))
    bodies=[model.body(i).name for i in range(model.nbody)];joints=[model.joint(i).name for i in range(model.njnt)]
    old_bodies=[control.body(i).name for i in range(control.nbody)]
    object_bid=canonical_id(bodies,'wooden_spoon_object');old_bid=canonical_id(old_bodies,'wooden_spoon_object')
    def local_vertices(m,g):
        mesh=int(m.geom_dataid[g]);adr=int(m.mesh_vertadr[mesh]);num=int(m.mesh_vertnum[mesh])
        return np.einsum('ni,ji->nj',m.mesh_vert[adr:adr+num],Rotation.from_quat(m.geom_quat[g][[1,2,3,0]]).as_matrix(),optimize=False)+m.geom_pos[g]
    scene=compiled[0];newton_model=scene.model
    import newton
    body=int(scene.objects[0].root.body_id)
    shape_bodies=newton_model.shape_body.numpy();shape_flags=newton_model.shape_flags.numpy()
    shape_scale=newton_model.shape_scale.numpy();shape_pose=newton_model.shape_transform.numpy()
    expected=[]
    for index,b in enumerate(shape_bodies):
        if b!=body or not (int(shape_flags[index])&int(newton.ShapeFlags.COLLIDE_SHAPES)):continue
        source=newton_model.shape_source[index];assert source is not None
        local=np.asarray(source.vertices,dtype=float)*shape_scale[index]
        rotation=Rotation.from_quat(shape_pose[index,3:7]).as_matrix()
        expected.append(np.einsum('ni,ji->nj',local,rotation,optimize=False)+shape_pose[index,:3])
    actual=[local_vertices(model,g) for g,b in enumerate(model.geom_bodyid) if b==object_bid]
    assert len(actual)==len(expected)>0
    from scipy.optimize import linear_sum_assignment
    costs=np.array([[max(cKDTree(e).query(a)[0].max(),cKDTree(a).query(e)[0].max()) for a in actual] for e in expected])
    er,ar=linear_sum_assignment(costs);maximum=float(costs[er,ar].max())
    assert maximum<=1e-6,'Native scaled hulls differ from upstream body-local scaled geometry'
    report['native_scaled_hull_import_checks']=dict(upstream_convex_parts=len(expected),native_convex_parts=len(actual),maximum_symmetric_vertex_error_m=maximum,passed=True,comparison='Actual Newton source hulls with their recorded body-local transforms/scales versus fresh native imported vertices')
    report['native_scaled_hulls_match_effective_geometry']=True
    report['new_cache_receipt_sha256']=sha(inputs/'collision-cache-receipt.json')
    roundtrip=trimesh.load(eval_path,force='mesh',process=False)
    np.testing.assert_allclose(roundtrip.vertices,np.asarray(source_mesh.vertices)*scale,atol=1e-8,rtol=0)
    with np.load(inputs/'targets.npz',allow_pickle=False) as z:q=z['robot_qpos'].copy();obj=z['object_poses'].copy()
    with np.load(S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz',allow_pickle=False) as z:camera=z['camera_c2w'].copy();flags=z['object_0_valid'].copy()
    stage=np.asarray(json.loads((inputs/'command-proposal-report.json').read_text())['stage_rotation_from_inferred_support_normal']);alignment=np.eye(4);alignment[:3,:3]=stage
    bindings={}
    for si,side in enumerate(['left','right']):
        bindings[side]=dict(si=si,wrist=[canonical_id(joints,f'{side}_wrist_{a}') for a in ['x','y','z','rotvec']],fingers=[canonical_id(joints,n) for n in manifest['joint_names'][side]],bodies={b for b,name in enumerate(bodies) if name.startswith('dex3_'+side+'_')},kin=URDFKinematics(inputs/f'robot/urdfs/g1/dex3_{side}.urdf'))
    support_bid=canonical_id(bodies,'support');adr=int(model.jnt_qposadr[model.body_jntadr[object_bid]])
    for row in sorted(fit['rows'],key=lambda r:r['frame']):
        frame=row['frame'];T=alignment@camera[frame]@np.asarray(row['candidate_camera_from_object'])
        state=mujoco.MjData(model);state.qpos[adr:adr+3]=T[:3,3];state.qpos[adr+3:adr+7]=Rotation.from_matrix(T[:3,:3]).as_quat()[[3,0,1,2]]
        for side,h in bindings.items():
            command=q[frame,h['si']]
            for col,j in enumerate(h['wrist'][:3]):state.qpos[int(model.jnt_qposadr[j])]=command[col]
            a=int(model.jnt_qposadr[h['wrist'][3]]);state.qpos[a:a+4]=command[[6,3,4,5]]
            for j,v in zip(h['fingers'],command[7:],strict=True):state.qpos[int(model.jnt_qposadr[j])]=v
        mujoco.mj_forward(model,state)
        max_fk=0.
        for side,h in bindings.items():
            command=q[frame,h['si']];poses=h['kin'].poses(command[:7],dict(zip(manifest['joint_names'][side],command[7:],strict=True)))
            for bid in h['bodies']:
                matches=[name for name in poses if bodies[bid]==name or bodies[bid].endswith('_'+name)];assert len(matches)==1
                p=poses[matches[0]];error=max(float(abs(state.xpos[bid]-p[:3,3]).max()),float(abs(state.xmat[bid].reshape(3,3)-p[:3,:3]).max()));assert error<2e-6;max_fk=max(max_fk,error)
        penetration=dict(object_support=0.,left_object=0.,left_support=0.,right_object=0.,right_support=0.)
        for contact in state.contact[:state.ncon]:
            a,b=[int(model.geom_bodyid[g]) for g in [contact.geom1,contact.geom2]];pair={a,b};depth=max(0.,-float(contact.dist))
            if pair=={object_bid,support_bid}:penetration['object_support']=max(penetration['object_support'],depth)
            for side,h in bindings.items():
                for target,name in [(object_bid,'object'),(support_bid,'support')]:
                    if (a in h['bodies'] and b==target) or (b in h['bodies'] and a==target):penetration[f'{side}_{name}']=max(penetration[f'{side}_{name}'],depth)
        report['frames'].append(dict(frame=frame,fit_accepted=row['fit']['accepted'],learned_depth_gate_passed=row['learned_depth_gate_rejection'] is None,retained_sgbm_depth_gate_passed=row['retained_sgbm_depth_gate_rejection'] is None,original_object_observed_flag=bool(flags[frame]),maximum_native_penetration_by_pair_m=penetration,all_hand_fk_verified=True,maximum_fk_error=max_fk,object_support_sample_gate_passed=penetration['object_support']<=.001,all_hand_target_sample_gate_passed=max(penetration.values())<=.001))
        save()
    report['all_seven_object_support_samples_passed']=all(v['object_support_sample_gate_passed'] for v in report['frames'])
    report['all_seven_full_target_contact_samples_passed']=all(v['all_hand_target_sample_gate_passed'] for v in report['frames'])
    for name,h in manifest['files'].items():assert sha(inputs/name)==h
    previous=json.loads((P/'manifest.json').read_text())
    for name,h in previous['files'].items():assert sha(P/name)==h
    report['frozen_original_and_candidate_closures_verified_after']=True
except BaseException as error:report['error']=repr(error);raise
finally:report['wall_seconds']=time.monotonic()-started;save();print(json.dumps({k:v for k,v in report.items() if k!='native_scaled_hull_import_checks'},indent=2),flush=True)
