"""Independent source-mask, complex-chain derivatives and exact angle-row binding.

No producer/shared FK/old polynomial or model audit is called. No optimizer.
"""
from pathlib import Path
from fractions import Fraction as F
import cmath,datetime,hashlib,json,math,os,time,xml.etree.ElementTree as ET
import numpy as np
from thumb_conic_geometry_v1 import plain,journal
S=Path('/Users/rohithiruku/v2d_scored_baseline_20261017');P=S/'e037-coupled-angle-model-v1';O=S/'e037-coupled-angle-binding-v1';FROZEN=S/'e037-thumb-ray-independent-validation-v1/frozen';start=time.monotonic();read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dec(v):return F(int(v['numerator']),int(v['denominator']))
def save(n,v):
 p=O/n;p.write_text(json.dumps(plain(v),indent=2,allow_nan=False)+'\n');p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return p
def mm(A,B):return [[sum(A[i][k]*B[k][j]for k in range(len(B)))for j in range(len(B[0]))]for i in range(len(A))]
def eye():return [[complex(i==j)for j in range(4)]for i in range(4)]
def transform(R,p):
 T=eye()
 for i in range(3):T[i][:3]=R[i];T[i][3]=p[i]
 return T
def axis_rotation(a,theta):
 c=cmath.cos(theta);s=cmath.sin(theta);cross=[[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]]
 return [[c*(i==j)+(1-c)*a[i]*a[j]+s*cross[i][j]for j in range(3)]for i in range(3)]
def root_rotation(q):
 norm=math.sqrt(math.fsum(v*v for v in q));x,y,z,w=[v/norm for v in q]
 return [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
model=read(P/'fullclock-coupled-angle-model-binding.json')
for n,d in model['input_sha256'].items():assert sha(n)==d
for n,k in [('continuous-angle-variables.json','variables_sha256'),('all-exact-linear-angle-inequalities.json','linear_rows_sha256'),('all-direct-source-factor-proposals.json','direct_factors_sha256'),('actual-mask-and-immutable-source-bindings.json','mask_binding_sha256'),('retained-halfangle-domains-and-unknown-source-evidence.json','retained_domain_sha256'),('unqualified-model-fixed-components-and-initial-F.npz','model_arrays_sha256')]:assert sha(P/n)==model[k]
manifest=read(FROZEN/'manifest.json');names=manifest['joint_names']['right'];selected=[names[j]for j in[0,1,4,5,6]];vars=read(P/'continuous-angle-variables.json');rows=read(P/'all-exact-linear-angle-inequalities.json');factors=read(P/'all-direct-source-factor-proposals.json');mask=read(P/'actual-mask-and-immutable-source-bindings.json');MIDDLE=S/'e037-saved-middle-source-component-v1/UNQUALIFIED-middle-source-motion-component.npz';WRIST=S/'e037-strict-source-wrist-mode-v1/UNQUALIFIED-strict-source-root-proposal.npz'
with np.load(FROZEN/'targets.npz')as z:q=z['robot_qpos'];ts=z['timestamps_s']
with np.load(FROZEN/'aligned-source-reference.npz')as z:original=z['robot_qpos'];human=z['hand_joints_world'];cameras=z['camera_c2w']
with np.load(FROZEN/'source-confidence.npz')as z:valid=z['hand_valid']
with np.load(S/'e037-right-late-independent-validation-v1/frozen/targets.npz')as z:prior=z['robot_qpos']
with np.load(S/'e037-left-thumb-independent-validation-v1/frozen/targets.npz')as z:left=z['robot_qpos']
with np.load(MIDDLE)as z:middle=z['robot_qpos']
with np.load(WRIST)as z:wrist=z['robot_qpos']
with np.load(P/'unqualified-model-fixed-components-and-initial-F.npz')as z:base=z['model_base_qpos'];initial=z['initial_F_variables'];assert np.array_equal(z['original_F_qpos'],q)and np.array_equal(z['original_aligned_qpos'],original)and np.array_equal(z['timestamps_s'],ts)
assert np.array_equal(base[:,0],q[:,0])and np.array_equal(base[:,:,:7],wrist[:,:,:7])and np.array_equal(base[:,1,9:11],middle[:,1,9:11])and np.array_equal(base[:,1,7:9],q[:,1,7:9])and np.array_equal(base[:,:,11:],q[:,:,11:])
immutable=set(np.flatnonzero(np.any(q[:,1,11:14]!=prior[:,1,11:14],axis=1)).tolist())|set(np.flatnonzero(np.any(prior[:,1,11:14]!=left[:,1,11:14],axis=1)).tolist());assert immutable==set(mask['immutable_thumb_frames'])
stage=np.asarray(read(FROZEN/'command-proposal-report.json')['stage_rotation_from_inferred_support_normal'])
with np.load(S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz')as z:K=z['intrinsics']
caps={r['frame']:r for r in read(FROZEN.parent/'report.json')['source_rows']if r['hand']=='right'};U=FROZEN/'robot/urdfs/g1/dex3_right.urdf';xml=ET.parse(U).getroot();jointnodes=list(xml.findall('joint'));child={j.find('child').attrib['link']:j for j in jointnodes};rootlink=next(iter({p.attrib['name']for p in xml.findall('link')}-set(child)));limits={j.attrib['name']:j.find('limit')for j in jointnodes};free=set();availability={};expectedfactors=set();columns={v['name']:v['column']for v in vars};certs=[]
J=(O/'normalized-BEFORE-AFTER-journal.jsonl').open('w');emit=lambda v:journal(J,v);emit(dict(kind='BEFORE_NEW_independent_model_binding',model_binding_sha256=sha(P/'fullclock-coupled-angle-model-binding.json'),no_model_or_source_producer_or_old_audit_or_solver_replayed=True))
for r in mask['availability']:
 f=r['frame'];hc=(human[f,1,r['landmark']]-cameras[f,:3,3])@cameras[f,:3,:3];available=bool(valid[f,1]and np.isfinite(hc).all()and hc[2]>.01);assert available==r['original_available']==caps[f]['image_anchor_available'][[0,4,8,12].index(r['landmark'])];availability[(f,r['link'])]=available
 if available and r['link']in ['right_index_tip','right_thumb_tip']and(r['link']=='right_index_tip'or f not in immutable):
  expectedfactors.add((f,r['link']))
  for n in(names[:2]if r['link']=='right_index_tip'else names[4:]):free.add((f,n))
assert len(free)==465 and len(expectedfactors)==194 and len(availability)==604 and set((v['frame'],v['URDF_joint'])for v in vars)==free
for v in vars:
 assert v['name']==f'q_{v["frame"]}_{v["URDF_joint"]}'and names[v['finger_index']]==v['URDF_joint']and v['initial_F_angle_rad']==q[v['frame'],1,v['command_column']]==initial[v['column']]and columns[v['name']]==v['column']
assert set((f['frame'],f['link'])for f in factors)==expectedfactors
save('independent-availability-preservation-column-bindings.json',dict(availability_bindings=604,variables=465,source_factors=194,immutable_thumb_frames=sorted(immutable),all_initial_F_values_and_original_missing_commands_UNCHANGED=True,entire_left_and_all_immutable_thumb_commands_exact=True,root_and_middle_model_arrays_match_saved_components_exactly=True))
sourcepolypath=S/'e037-index-source-polynomial-v2/actual-index-polynomials-and-saved-region-charts.json';region=next(r for r in read(sourcepolypath)['regions']if r['saved_node_id']==2);box=np.asarray(region['box']);termcount=0;constantcount=0
for row in rows:
 emit(dict(kind='BEFORE_NEW_independent_angle_row',row_id=row['id'],actual_row=row));n=row['joint'];j=names.index(n);t=row['terms'][0]['frame'];kind=row['kind'];tol=F(float(1e-7));ptol=F(float(1e-8));cap=F(float(.25))+tol;change=F(float(.05))+tol;ref=lambda f:F(float(original[f,1,7+j]));unit=F(1)
 if kind=='actual_URDF_position_upper':terms=[(t,unit)];upper=F(float(limits[n].attrib['upper']))+ptol
 elif kind=='actual_URDF_position_lower':terms=[(t,-unit)];upper=-F(float(limits[n].attrib['lower']))+ptol
 elif kind=='original_component_upper':terms=[(t,unit)];upper=ref(t)+cap
 elif kind=='original_component_lower':terms=[(t,-unit)];upper=-ref(t)+cap
 elif kind=='original_adjacent_added':
  sign=row['origin']['sign'];terms=[(t,F(sign)),(t-1,F(-sign))];upper=sign*(ref(t)-ref(t-1))+change
 elif kind=='original_second_added':
  sign=row['origin']['sign'];terms=[(t,F(sign)),(t-1,F(-2*sign)),(t-2,F(sign))];upper=sign*(ref(t)-2*ref(t-1)+ref(t-2))+change
 elif kind=='actual_URDF_velocity':
  sign=row['origin']['sign'];terms=[(t,F(sign)),(t-1,F(-sign))];dt=F(float(ts[t]))-F(float(ts[t-1]));assert dt==F(float(ts[t]-ts[t-1]));upper=(F(float(limits[n].attrib['velocity']))+tol)*dt
 elif kind=='inherited_region2_index_lower':terms=[(t,-unit)];upper=-F(float(box[t,j,0]))
 elif kind=='inherited_region2_index_upper':terms=[(t,unit)];upper=F(float(box[t,j,1]))
 else:raise ValueError('unknown new linear gate')
 assert [(r['frame'],dec(r['coefficient']))for r in row['terms']]==terms and dec(row['upper'])==upper
 constant=sum((a*F(float(q[f,1,7+j]))for f,a in terms if(f,n)not in free),F(0));active=[(f'q_{f}_{n}',f,a)for f,a in terms if(f,n)in free];actual=[(r['variable'],r['frame'],dec(r['coefficient']))for r in row['active_terms']];assert actual==active and dec(row['fixed_sum'])==constant and dec(row['folded_upper'])==upper-constant and row['constant_only']==(not active)
 if not active:assert row['constant_only_passed']==(constant<=upper);constantcount+=1
 cert=dict(row_id=row['id'],kind=kind,exact_original_binary_expression_bound=True,exact_shared_column_aliases_and_fixed_folding=True,coefficient_count=len(terms),constant_only=not active);termcount+=len(terms);certs.append(cert);emit(dict(kind='AFTER_NEW_independent_angle_row',returned=cert))
save('all-independent-original-angle-row-certificates.json',certs)
def complex_point(rootpose,qvalues,link):
 chain=[];cur=link
 while cur!=rootlink:j=child[cur];chain.append(j);cur=j.find('parent').attrib['link']
 T=transform(root_rotation(rootpose[3:]),rootpose[:3]);positions=dict(zip(names,qvalues,strict=True))
 for node in reversed(chain):
  origin=node.find('origin');p=[0.,0.,0.]if origin is None else list(map(float,origin.attrib.get('xyz','0 0 0').split()));rpy=[0.,0.,0.]if origin is None else list(map(float,origin.attrib.get('rpy','0 0 0').split()));R=mm(mm(axis_rotation([0,0,1],rpy[2]),axis_rotation([0,1,0],rpy[1])),axis_rotation([1,0,0],rpy[0]));T=mm(T,transform(R,p))
  if node.attrib['type']=='revolute':T=mm(T,transform(axis_rotation(list(map(float,node.find('axis').attrib['xyz'].split())),positions[node.attrib['name']]),[0,0,0]))
 return np.array([T[i][3]for i in range(3)],complex)
def independent_factor(factor,qvalues):
 f=factor['frame'];p=complex_point(wrist[f,1,:7],qvalues,factor['link']);c=(p@stage-cameras[f,:3,3])@cameras[f,:3,:3];hc=(human[f,1,factor['landmark']]-cameras[f,:3,3])@cameras[f,:3,:3];pixel=np.array([K[0,0]*hc[0]/hc[2]+K[0,2],K[1,1]*hc[1]/hc[2]+K[1,2]]);radius=caps[f]['source_before_after_px'][0][factor['source_tip_index']]+1e-6;X=K[0,0]*c[0]+(K[0,2]-pixel[0])*c[2];Y=K[1,1]*c[1]+(K[1,2]-pixel[1])*c[2];g=radius**2*c[2]**2-X**2-Y**2
 return p,c,g,c[2]-.01,pixel
worlderr=cameraerr=gerr=worldJerr=sourceJerr=depthJerr=0.;sourcecerts=[];h=1e-20
for factor in factors:
 f=factor['frame'];emit(dict(kind='BEFORE_NEW_independent_complex_source_derivative',factor_id=factor['id'],frame=f,link=factor['link']));actual=read(P/f'factor-{factor["id"]:03d}-actual-constructor-return.json');assert actual['factor']==factor and np.array_equal(factor['root_pose'],wrist[f,1,:7])and np.array_equal(factor['fixed_F_full_finger_values'],base[f,1,7:])and factor['ORIGINAL_cap_px']==caps[f]['source_before_after_px'][0][factor['source_tip_index']]
 values=np.asarray(base[f,1,7:],complex);p,c,g,d,pixel=independent_factor(factor,values);R=actual['initial_F_source_RETURN'];FK=actual['initial_F_chain_RETURN'];p0=np.array(FK['point']);J0=np.array(FK['jacobian']);C0=np.array(R['camera_point']);g0=R['source_cone_value'];derivatives=[];indJ=np.zeros((3,7));gJ=np.zeros(7);dJ=np.zeros(7)
 for j in range(7):
  x=values.copy();x[j]+=1j*h;pj,cj,gj,dj,_=independent_factor(factor,x);indJ[:,j]=pj.imag/h;gJ[j]=gj.imag/h;dJ[j]=dj.imag/h
  derivatives.append(dict(joint=names[j],point_derivative=indJ[:,j].copy(),source_derivative=gJ[j],depth_derivative=dJ[j]))
 pe=float(abs(p.real-p0).max());ce=float(abs(c.real-C0).max());ge=abs(float(g.real)-g0);je=float(abs(indJ-J0).max());se=float(abs(gJ-np.array(R['source_cone_jacobian'])).max());de=float(abs(dJ-np.array(R['strict_depth_jacobian'])).max());worlderr=max(worlderr,pe);cameraerr=max(cameraerr,ce);gerr=max(gerr,ge);worldJerr=max(worldJerr,je);sourceJerr=max(sourceJerr,se);depthJerr=max(depthJerr,de)
 certificate=dict(factor_id=factor['id'],world_error_m=pe,camera_error_m=ce,source_squared_cone_error=ge,world_J_error=je,source_J_error=se,depth_J_error=de,all_7_complex_chain_derivatives=derivatives,diagnostic_envelopes_NOT_original_candidate_tolerances=True);save(f'factor-{factor["id"]:03d}-independent-derivative-certificate.json',certificate);emit(dict(kind='AFTER_NEW_independent_complex_source_derivative',returned=certificate))
 assert pe<1e-12 and ce<1e-12 and je<1e-12 and de<1e-12 and ge<2e-9*max(1.,abs(g0))and se<2e-9*max(1.,float(abs(np.array(R['source_cone_jacobian'])).max()))and np.max(abs(pixel-factor['source_pixel']))<1e-12
 assert [(v['variable'],v['column'],v['finger_index'])for v in actual['active_columns']]==[(f'q_{f}_{n}',columns[f'q_{f}_{n}'],names.index(n))for n in factor['free_joint_names']]
 sourcecerts.append(certificate)
domains=read(P/'retained-halfangle-domains-and-unknown-source-evidence.json');old=read(S/'e037-surrounding-clock-model-v1/surrounding-model-binding.json');assert domains['full_region2_COPIED']==old['full_region2_COPIED']and domains['three_unknown109bands_COPIED']==old['three_unknown109bands_COPIED']
for r,v in zip(domains['retained_domain_factors'],old['variables'],strict=True):
 assert r['half_angle_name']==v['name']and r['angle_bounds_COPIED']==v['angle_bounds']and r['half_angle_bounds_COPIED']==v['half_angle_bounds']and r['origin_COPIED']==v['origin']and r['finite_angle_chart']and all(abs(x['approx'])<3 for x in r['angle_bounds_COPIED'])
save('independent-retained-domains-and-numerical-scope.json',dict(exact_domain_copy_identities=38,full_region2_and_all_unknown109bands_exactly_retained=True,no_endpoint_trig_or_source_polynomial_recomputed=True,exact_linear_arithmetic_scope='exact real expressions over saved binary angles/timestamps/caps/tolerances; this does not reproduce every intermediate floating arithmetic rounding',all_original_direct_candidate_tolerances_UNCHANGED=True,no_original_gate_is_accepted_by_a_new_derivative_or_rounding_envelope=True))
summary=dict(schema='v2d.coupled-angle-independent-model-binding.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_NEW_mask_column_linear_chain_source_derivative_bindings_passed=True,source_mask_decisions=604,continuous_variable_bindings=465,exact_original_angle_row_bindings=len(rows),exact_linear_term_identities=termcount,constant_only_rows_verified=constantcount,actual_URDF_source_factor_bindings=len(sourcecerts),independent_complex_joint_derivatives=len(sourcecerts)*7,world_J_coefficients=len(sourcecerts)*21,retained_domain_copy_identities=38,maximum_world_closure_m=worlderr,maximum_camera_closure_m=cameraerr,maximum_source_cone_closure=gerr,maximum_world_J_difference=worldJerr,maximum_source_J_difference=sourceJerr,maximum_depth_J_difference=depthJerr,new_model_builder_calls=0,new_optimizer_or_solver_calls=0,new_candidate_points=0,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,source_authority_F_components_and_original_gates_UNCHANGED=True,wall_seconds=time.monotonic()-start);emit(dict(kind='AFTER_NEW_independent_model_binding',returned=summary));J.close();summary['all_outputs_sha256']={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log','report.json']};save('report.json',summary);print(json.dumps({k:v for k,v in summary.items()if k!='all_outputs_sha256'}))
