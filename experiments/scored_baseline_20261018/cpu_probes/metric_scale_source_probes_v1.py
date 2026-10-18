"""Three identical CPU shape-scale probes; preserve all baseline meshes/poses."""
from pathlib import Path
import hashlib,json,subprocess,sys,time
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');OUT=S/'metric-scale-source-probes-v1';OUT.mkdir();rows=[];started=time.monotonic()
for ep,cid in [(33,13),(35,13),(37,14)]:
 source=S/f'scored-cohort-{cid:02d}-results/reconstruction-e{ep:03d}/geometry/e{ep:03d}/wooden_spoon';dest=OUT/f'e{ep:03d}';before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()}
 with (OUT/f'e{ep:03d}.log').open('wb') as log:r=subprocess.run([sys.executable,str(R/'scripts/track3/reproduce_metric_seed_candidate.py'),'--source',str(source),'--out',str(dest)],stdout=log,stderr=subprocess.STDOUT,timeout=90)
 if before!={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()}:raise ValueError('Source shape or measured evidence changed')
 row=dict(episode=ep,returncode=r.returncode,source_files_unchanged=True)
 if r.returncode==0:
  candidate=json.loads((dest/'candidate-report.json').read_text());row.update({k:candidate[k] for k in ['uniform_scale','anchor_translation_delta_m','before','after','optimizer_success']});row['source_fit_improved']=candidate['after']['quality']>candidate['before']['quality'] and candidate['after']['silhouette_iou']>=candidate['before']['silhouette_iou'] and candidate['after']['median_measured_vertex_distance_m']<candidate['before']['median_measured_vertex_distance_m']
 rows.append(row);print(json.dumps(row),flush=True)
report=dict(schema='v2d.metric-scale-source-probes.v1',rows=rows,same_fit_bounds_and_source_gates_for_all_cases=True,source_seed_fit_only=True,multiframe_generalization_assessed=False,physics_stepped=False,baseline_inputs_changed=False,candidates_installed=False,new_gpu_lease_started=False,physical_task_accepted=False,wall_seconds=time.monotonic()-started)
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
