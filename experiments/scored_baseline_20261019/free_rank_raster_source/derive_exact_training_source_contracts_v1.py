"""NEW free-rank/union/raster contracts; synthetic controls only, no episode."""
from pathlib import Path
from fractions import Fraction as F
import datetime, hashlib, itertools, json, math, os
import cv2, numpy as np
from exact_training_source_logic_v1 import (free_rank_formula,rank_event,
 cardinality_truth,nearest_union_event,quantized_interval,positive_projection,
 silhouette_count_envelope,triangle_line8_integer,line8_pixels,clipped_line)

S=Path(__file__).resolve().parents[1]; O=S/'e037-exact-training-source-contracts-v1'
def plain(v):
 if isinstance(v,F):return {'numerator':str(v.numerator),'denominator':str(v.denominator)}
 if isinstance(v,np.ndarray):return plain(v.tolist())
 if isinstance(v,np.generic):return plain(v.item())
 if isinstance(v,dict):return {str(k):plain(x)for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [plain(x)for x in v]
 if isinstance(v,float)and not math.isfinite(v):return {'nonfinite_float':repr(v)}
 return v
def save(name,value):
 p=O/name;p.write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n');p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

with(O/'normalized-BEFORE-AFTER-journal.jsonl').open('w')as log:
 def emit(v):
  log.write(json.dumps(plain(v),allow_nan=False)+'\n');log.flush();os.fsync(log.fileno())
 controls=[]
 def control(name,constructor,evaluate,check):
  before=save(name+'-BEFORE.json',constructor);emit(dict(kind='BEFORE_NEW_contract',name=name,constructor=before))
  ret=evaluate();after=save(name+'-RETURN.json',ret);emit(dict(kind='AFTER_NEW_actual_RETURN',name=name,actual_RETURN=after))
  assert check(ret), name
  controls.append(name)
 declaration=dict(schema='v2d.NEW-exact-training-source-contracts.v1',
  no_episode_data_or_pose_operator_calls=True,no_old_control_replay=True,
  free_thresholds_subsets_and_all_original_visual_vertex_choices=True,
  exact_rational_ideal_distance_logic_SEPARATE_from_actual_binary64_KD_norm_median=True,
  raster='NEW source-informed LINE_8 shift0 interpreter; public 4.11.0 source tag alone is not installed build equivalence proof',
  full_source_visual_triangle_union_NOT_front_surface_or_z_buffer=True,
  unresolved_quantizer_KD_norm_build_and_source_uncertainty_NOT_relabelled=True,
  original_fullpixel_heldout_source_native_and_motion_gates_UNCHANGED=True)
 save('ONE-NEW-contract-declaration-BEFORE.json',declaration);emit(dict(kind='BEFORE_NEW_contracts',declaration=declaration))
 # General proof has no selected sample/rank/vertex subset: <= a at rank lo and
 # <= b at rank hi imply both order bounds. Conversely choose actual rank values.
 proof=dict(even='n=2k. If count(d<=a)>=k, count(d<=b)>=k+1, a<=b and a+b<=2m, then d_(k-1)+d_(k)<=a+b<=2m. Conversely choose a=d_(k-1),b=d_(k).',
  odd='n=2k+1. count(d<=a)>=k+1 and a<=m iff d_(k)<=m, choosing a=d_(k) in the reverse direction.',
  ties='Counts use <=. Neither subsets nor nearest-vertex choices are frozen.',
  floating='For finite nonnegative returned binary64 distances, fl(fl(a+b)/2) is monotone in each ordered double threshold. The same cardinality proof applies with this ORIGINAL rounded operator; it is NOT silently replaced by a real-valued sum inequality.',
  nearest='For a FINITE nonempty vertex set and ideal Euclidean distance, min_j norm(l-v_j)<=r iff OR_j sum_k(l_k-v_jk)^2<=r^2 for r>=0. This rational polynomial identity does not certify the floating KD distance algorithm.',
  raster='After strict depth and ORIGINAL clip/truncation, the full silhouette is OR over the LINE_8 triangle programs, followed by EVEN-domain counts; overlapping or hidden triangles are not dropped.',
  count_bounds='For C subset actual P subset Q, I_low=|C and M and D|, I_high=|Q and M and D|, U_low=|M and D|+|C and not M and D|, U_high similarly Q. IoU determinant lies between I_low U0-I0 U_high and I_high U0-I0 U_low.')
 save('NEW-general-equivalence-and-outward-count-proofs.json',proof)
 control('NEW-free-even-threshold-compensation',dict(d=[0,3,5,9],m=4,old_fixed_radii=[2,6]),
  lambda:dict(free=rank_event([0,3,5,9],4,3,5),old_fixed=rank_event([0,3,5,9],4,2,6)),
  lambda r:r['free']['accepted']and not r['old_fixed']['accepted'])
 cases=[[F(0),F(1),F(4),F(8)], [F(1),F(1),F(1),F(1),F(4),F(7)], [F(0),F(0)]]
 def rational_returns():
  out=[]
  for ds in cases:
   ordered=sorted(ds);lo=(len(ds)-1)//2;hi=len(ds)//2;m=(ordered[lo]+ordered[hi])/2
   out.append(dict(formula=free_rank_formula(len(ds),m),exact_median=m,
    boundary=rank_event(ds,m,ordered[lo],ordered[hi]),
    below=rank_event(ds,m-F(1,2**80),ordered[lo],ordered[hi])))
  return out
 control('NEW-even-free-rank-necessity-sufficiency-and-ties',dict(cases=cases,proof=proof['even']),rational_returns,
  lambda rs:all(r['boundary']['accepted']and not r['below']['accepted']for r in rs))
 a=1.;b=float(np.nextafter(1.,math.inf))
 control('NEW-binary64-rounded-average-is-not-real-average',dict(a=a,b=b,m0=1.),
  lambda:dict(actual=rank_event([a,b],1.,a,b,floating=True),
   rational=rank_event([a,b],F(1),a,b),exact_average=(F(a)+F(b))/2),
  lambda r:r['actual']['accepted']and not r['rational']['accepted']and r['exact_average']>1)
 control('NEW-odd-free-threshold-no-extra-rank',dict(distances=[7,2,0,2,9],m=2),
  lambda:dict(formula=free_rank_formula(5,2),witness=rank_event([7,2,0,2,9],2,2)),
  lambda r:r['formula']['threshold_symbols']==['a']and r['witness']['accepted'])
 control('NEW-zero-and-subnormal-rounded-rank-events',dict(zero=[0,0,0,7],subnormal=float(np.nextafter(0.,1.))),
  lambda:dict(zero=rank_event([0,0,0,7],0,0,0),
   subnormal=rank_event([0.,float(np.nextafter(0.,1.))],0.,0.,float(np.nextafter(0.,1.)),floating=True),
   rational=rank_event([F(0),F(1,2**1074)],0,0,F(1,2**1074))),
  lambda r:r['zero']['accepted']and r['subnormal']['accepted']and not r['rational']['accepted'])
 control('NEW-missing-cardinality-retained-and-negative-invalid',dict(events=[True,None,False],requirements=[1,2,3]),
  lambda:dict(counts=[cardinality_truth([True,None,False],i)for i in [1,2,3]],
   empty=free_rank_formula(0,None),missing=rank_event([0,None],1,0,1),
   invalid=rank_event([-1,0],1,0,0)),
  lambda r:[x['status']for x in r['counts']]==['PASS','UNRESOLVED','FAIL']and not r['missing']['accepted']and not r['invalid']['accepted'])
 V=[[0,0,0],[2,0,0],[0,3,0]]
 control('NEW-nearest-vertex-union-keeps-changing-witness',dict(vertices=V,points=[[F(1,4),0,0],[F(7,4),0,0]],radius=F(1,2)),
  lambda:dict(unions=[nearest_union_event(p,V,F(1,2))for p in [[F(1,4),0,0],[F(7,4),0,0]]],
   witness_ids=[0,1],missing=nearest_union_event(None,V,1),negative=nearest_union_event([0,0,0],V,-1),
   zero=nearest_union_event([2,0,0],V,0)),
  lambda r:all(r['unions'])and r['missing']is None and r['negative']is False and r['zero'])
 # Fractional linear coordinates deliberately do not assume a rounded rotation
 # has exactly R^T R=I; retain both matrices in the alias.
 R=[[F(1),F(1,8),0],[0,F(1),0],[0,0,F(1)]];W=[[0,-1,0],[1,0,0],[0,0,1]]
 P=[F(3,2),F(5,4),2];t=[F(1,4),F(1,2),1];d=[F(1,8),-F(1,4),F(1,2)]
 def alias():
  B=[[-sum(R[k][i]*W[k][j]for k in range(3))for j in range(3)]for i in range(3)]
  left=[sum((P[k]-t[k]-sum(W[k][j]*d[j]for j in range(3)))*R[k][i]for k in range(3))for i in range(3)]
  right=[sum((P[k]-t[k])*R[k][i]for k in range(3))+sum(B[i][j]*d[j]for j in range(3))for i in range(3)]
  return dict(B=B,direct=left,affine=right)
 control('NEW-free-union-world-local-axis-alias',dict(R=R,Rcw=W,P=P,t=t,world_delta=d),alias,
  lambda r:r['direct']==r['affine'])
 control('NEW-strict-depth-parallel-and-missing-projections',dict(depths=[F(1,100),0,-1,1]),
  lambda:[positive_projection([1,2,z],4,5,6,7)for z in [F(1,100),0,-1,1]]+[positive_projection(None,4,5,6,7)],
  lambda r:all(x['status'].startswith('UNRESOLVED')for x in r[:3]+r[4:])and r[3]['uv']==[10,17])
 qcases=[(-F(19,10),-F(11,10)),(-F(11,10),-F(9,10)),(-F(9,10),F(9,10)),(F(9,10),1),(20001,30000),(-30000,-20001)]
 control('NEW-clip-truncate-cell-and-boundary-enclosures',dict(intervals=qcases),
  lambda:[quantized_interval(*p)for p in qcases],
  lambda r:[x['status']for x in r]==['CERTAIN_integer','UNRESOLVED_integer_boundary','CERTAIN_integer','UNRESOLVED_integer_boundary','CERTAIN_integer','CERTAIN_integer']and r[0]['minimum_integer']==-1 and r[-1]['maximum_integer']==-20000)
 # Only these NEW analytic triangles are rendered. No episode F baseline or
 # rejected point is fed to either implementation.
 triangles=[[[1,1],[9,2],[3,8]], [[-4,2],[7,-3],[10,9]],
            [[-20000,-20000],[20000,1],[1,20000]], [[3,3],[3,3],[3,3]],
            [[2,1],[2,8],[2,4]], [[-8,-4],[-2,-3],[-1,-1]]]
 def rasters():
  out=[]
  for k,v in enumerate(triangles):
   own=triangle_line8_integer(v,(11,13));actual=np.zeros((11,13),np.uint8)
   cv2.fillConvexPoly(actual,np.array(v,np.int32),255)
   p=O/f'NEW-synthetic-triangle-{k}-both-RETURN.npz';np.savez_compressed(p,independent=own,actual=actual,vertices=v);p.chmod(0o600)
   out.append(dict(case=k,returned_arrays=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
    differing_pixels=int(np.count_nonzero(own!=(actual>127))),coverage=int(np.count_nonzero(own))))
  return dict(installed_cv2=cv2.__version__,cases=out,finite_controls_NOT_full_build_equivalence=True)
 control('NEW-reference-LINE8-clipping-degenerate-branch-coverage',dict(shape=[11,13],triangles=triangles),rasters,
  lambda r:all(x['differing_pixels']==0 for x in r['cases']))
 lcases=[((-5,1),(20,8)),((2,-8),(8,20)),((10,1),(0,6)),((0,0),(8,4)),((-3,-3),(-1,-1))]
 def lines():
  out=[]
  for a,b in lcases:
   points=line8_pixels(a,b,13,11);own=np.zeros((11,13),np.uint8)
   for x,y in points:own[y,x]=255
   actual=np.zeros_like(own);cv2.line(actual,a,b,255,1,cv2.LINE_8)
   valid,c1,c2=cv2.clipLine((0,0,13,11),a,b);c=clipped_line(a,b,13,11)
   out.append(dict(points=points,clipping=c,actual_clipping=[bool(valid),c1,c2],
    pixel_error=int(np.count_nonzero(own!=actual)),clip_equal=(c==(tuple(c1),tuple(c2))if valid else c is None)))
  return out
 control('NEW-integer-line-tie-and-double-clipping-program',dict(shape=[11,13],lines=lcases),lines,
  lambda r:all(x['pixel_error']==0 and x['clip_equal']for x in r))
 def union():
  a=triangle_line8_integer(triangles[0],(11,13));b=triangle_line8_integer([[3,1],[11,8],[1,8]],(11,13))
  reverse=triangle_line8_integer(triangles[0][::-1],(11,13));u=a|b
  return dict(a=int(a.sum()),b=int(b.sum()),overlap=int((a&b).sum()),union=int(u.sum()),
   reversed_identical=bool(np.array_equal(a,reverse)),occluded_triangles_NOT_removed=True)
 control('NEW-full-triangle-union-and-orientation',dict(triangles=[triangles[0],[[3,1],[11,8],[1,8]]]),union,
  lambda r:r['union']==r['a']+r['b']-r['overlap']and r['reversed_identical']and r['overlap']>0)
 C=np.array([[1,0,0],[0,0,0]],bool);Q=np.array([[1,1,1],[0,1,0]],bool);M=np.array([[1,1,0],[0,0,0]],bool);D=np.ones_like(C)
 def outward_counts():
  envelope=silhouette_count_envelope(C,Q,M,D,1,2);unknown=list(zip(*np.nonzero(Q&~C)));values=[]
  for bits in itertools.product([False,True],repeat=len(unknown)):
   P=C.copy()
   for ij,b in zip(unknown,bits,strict=True):P[ij]=b
   I=int((P&M).sum());U=int((P|M).sum());values.append(I*2-U)
  return dict(envelope=envelope,all_possible_determinants=values,
   exact=silhouette_count_envelope(M,M,M,D,2,2),
   empty=silhouette_count_envelope(np.zeros_like(C),np.zeros_like(C),np.zeros_like(C),D,0,0))
 control('NEW-outward-discrete-IoU-count-with-unknown-pixels',dict(certain=C,possible=Q,mask=M,domain=D,baseline_I=1,baseline_U=2),outward_counts,
  lambda r:all(r['envelope']['determinant'][0]<=v<=r['envelope']['determinant'][1]for v in r['all_possible_determinants'])and r['envelope']['status']=='UNRESOLVED'and r['exact']['status']=='PASS'and r['empty']['status']=='UNRESOLVED_empty_union')
 # Combined real quality has no separate fitted weight: monotonicity in I/U and
 # median suffices. Actual floating quality remains the unmodified post-filter.
 control('NEW-combined-count-median-quality-monotonic-scope',dict(base_iou=F(2,5),base_median=F(1,50),new_iou=F(1,2),new_median=F(1,100)),
  lambda:dict(exact_iou_nonregression=F(1,2)>=F(2,5),exact_median_nonregression=F(1,100)<=F(1,50),
   analytic_log_quality_ratio_positive=True,proof='log((1/2)/(2/5))+(1/50-1/100)/(1/50)=log(5/4)+1/2>0',
   actual_binary64_quality_POSTcheck_REQUIRED=True),
  lambda r:r['exact_iou_nonregression']and r['exact_median_nonregression'])
 report=dict(schema='v2d.NEW-free-rank-union-raster-contracts.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  NEW_contracts=len(controls),contracts=controls,all_pass=True,installed_cv2=cv2.__version__,
  rational_equivalence_proofs_saved=True,actual_rounded_median_semantics_separate=True,
  no_episode_pose_source_raster_NN_solver_native_or_GPU_calls=True,
  source_reference_build_and_KD_boundary_errors_UNRESOLVED=True,
  no_original_gate_tolerance_radius_source_geometry_or_raw_return_change=True)
 save('report.json',report);emit(dict(kind='AFTER_NEW_contracts',actual_RETURN=report));print(json.dumps(report))
