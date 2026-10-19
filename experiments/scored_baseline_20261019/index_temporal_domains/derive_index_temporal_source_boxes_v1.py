"""NEW correlated 107/108 -> continuous 109 source/temporal restrictions.

All saved consecutive pairs form a conservative superset. No old graph,
source constructor or missing forward-pair record is recomputed.
"""
from pathlib import Path
import datetime, hashlib, itertools, json, sys, time
S = Path(__file__).resolve().parents[1]
O = S/'e037-index-temporal-source-boxes-v1'
sys.path.insert(0, str(S/'compute'))
import source_orientation_context_v1 as c
import thumb_conic_geometry_v1 as g
from index_temporal_box_v1 import next_correction_box
from closed_2R_source_domains_v1 import closed_2R_source_chart
from scipy.spatial.transform import Rotation
np = c.np
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
start = time.monotonic()
j = (O/'all-correlated-pair-proposals-source-box-results.jsonl').open('w')
controls = []
def check(name, pair, refs, dt, lo, hi, velocity, assertion):
    g.journal(j, dict(kind='BEFORE_new_box_control', name=name, pair=pair, refs=refs, dt=dt))
    result = next_correction_box(pair, refs, dt, lo, hi, velocity,
                                 motion_tolerance=0., position_tolerance=0.)
    g.journal(j, dict(kind='AFTER_new_box_control', name=name, result=result))
    passed = bool(assertion(result)); controls.append(dict(name=name, passed=passed))
    assert passed, name
check('adjacent_and_second_intersection', [[0.], [.03]], [[0.], [0.], [0.]], [.1, .1], [-2.], [2.], [12.],
      lambda r: r['pair_passed_local_checks'] and np.allclose([r['joint_lower'][0],r['joint_upper'][0]], [.01,.08]))
check('actual_velocity_uses_actual_reference_jump', [[0.], [.03]], [[0.], [0.], [1.]], [.1, .01], [-2.], [2.], [1.],
      lambda r: r['pair_passed_local_checks'] and r['empty'])
check('bad_consecutive_pair_not_prefix_evidence', [[0.], [.06]], [[0.], [0.], [0.]], [.1, .1], [-2.], [2.], [12.],
      lambda r: not r['pair_passed_local_checks'] and not r['pair_checks']['adjacent_added'])
check('active_native_position_clips_outer_box', [[0.], [0.]], [[0.], [0.], [0.]], [.1, .1], [-.01], [.02], [12.],
      lambda r: np.allclose([r['joint_lower'][0],r['joint_upper'][0]], [-.01,.02]))

bank_path = S/'e037-integrated-source-corpora-reachability-v1/all-factorized-saved-banks-and-recurrence-returns.jsonl'
parent = json.loads((bank_path.parent/'report.json').read_text())
assert sha(bank_path) == parent['all_outputs_sha256'][bank_path.name]
banks = {}
for line in bank_path.read_text().splitlines():
    row = json.loads(line)
    if row.get('kind') == 'reused_all_source_points_without_Cartesian_pruning' and row.get('digit') == 'index' and row.get('frame') in [107,108]:
        banks[row['frame']] = np.array(row['bank'])
with np.load(S/'e037-strict-source-wrist-mode-v1/UNQUALIFIED-strict-source-root-proposal.npz') as z:
    commands = z['robot_qpos'].copy()
frame = 109
source_rows = {v['frame']:v for v in json.loads((S/'e037-thumb-ray-independent-validation-v1/report.json').read_text())['source_rows'] if v['hand']=='right'}
cap = source_rows[frame]['source_before_after_px'][0][2]
camera = c.cameras[frame]
pixel = c.uv((c.human[frame,1,[8]]-camera[:3,3])@camera[:3,:3])[0]
A = camera[:3,:3].T@c.stage.T
R = Rotation.from_quat(commands[frame,1,3:7]).as_matrix()
H = np.c_[A@R@np.eye(3)[:,:2], A@(commands[frame,1,:3]+R@np.array([.0777,-.0016,.0285]))-camera[:3,:3].T@camera[:3,3]]
records=[]; counts={}; unique_witnesses={}
for ia, ib in itertools.product(range(len(banks[107])), range(len(banks[108]))):
    pair=np.array([banks[107][ia],banks[108][ib]])
    before=dict(kind='BEFORE_new_correlated_index_box', pair_ids=[ia,ib], pair_frames=[107,108],
                pair=pair, original=c.aligned[107:110,1,7:9], dt=np.diff(c.ts[107:110]),
                source_frame=109, H=H, source_pixel=pixel, original_source_radius_px=cap,
                all_pairs_superset_NOT_actual_forward_mask=True)
    g.journal(j,before)
    result=next_correction_box(pair,c.aligned[107:110,1,7:9],np.diff(c.ts[107:110]),c.limits[:2,0],c.limits[:2,1],np.ones(2)*12)
    chart=None
    if not result['pair_passed_local_checks']:
        classification='rejected_consecutive_pair_local_motion'
    elif result['empty']:
        classification='empty_necessary_next_joint_box'
    else:
        box=np.c_[result['joint_lower'],result['joint_upper']]
        chart=closed_2R_source_chart(H,c.K,pixel,cap,box[0],box[1],image_tolerance=1e-6,position_tolerance=1e-8)
        classification=chart['classification']
        for w in chart['witnesses']:
            key=tuple(np.round(w['angles_rad'],12)); unique_witnesses.setdefault(key,dict(angles_rad=w['angles_rad'],pair_ids=[]))['pair_ids'].append([ia,ib])
    record=dict(pair_ids=[ia,ib],pair=pair,next_frame=109,box=result,classification=classification,source_chart=chart)
    g.journal(j,dict(kind='AFTER_new_correlated_index_box',record=record));records.append(record)
    counts[classification]=counts.get(classification,0)+1
j.close()
(O/'all-correlated-index-source-domains.json').write_text(json.dumps(g.plain(dict(
    source_frame=109,H=H,source_pixel=pixel,original_source_radius_px=cap,
    bank_path=str(bank_path),bank_sha256=sha(bank_path),records=records,
    unique_constructive_source_points=list(unique_witnesses.values()))),allow_nan=False,indent=2)+'\n')
report=dict(schema='v2d.index-temporal-source-boxes.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    producer_sha256=sha(__file__),input_manifest_sha256=sha(c.F/'manifest.json'),new_analytic_controls=controls,
    pair_bank_sizes=[len(banks[107]),len(banks[108])],total_correlated_pairs=len(records),classifications=counts,
    unique_constructive_source_points=len(unique_witnesses),saved_forward_pair_identities_not_reconstructed=True,
    all_pairs_are_conservative_superset_not_prefix_viability=True,each_pair_box_correlation_preserved=True,
    continuous_neighbors_and_full_path_unassessed=True,original_motion_tolerance_rad=1e-7,
    original_independent_position_tolerance_rad=1e-8,original_image_tolerance_px=1e-6,
    chart_absence_not_continuous_empty_proof=True,source_model_not_measured_robot_depth_or_contact=True,
    current_verified_reference_and_source_authority_unchanged=True,new_native_position_checks=0,
    new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,whole_reference_or_task_qualified=False,
    all_outputs_sha256={p.name:sha(p) for p in O.iterdir() if p.is_file() and p.name not in ['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(g.plain(report),allow_nan=False,indent=2)+'\n')
print(json.dumps(g.plain(report)))
