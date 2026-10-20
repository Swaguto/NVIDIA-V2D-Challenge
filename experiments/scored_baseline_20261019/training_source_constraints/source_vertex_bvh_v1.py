"""NEW independent radius-certificate BVH for source-visual-vertex nearest pairs.

This is a numerical identity check on saved coefficient models, not source
accuracy or a native convex-geometry certificate.
"""
import numpy as np

class SourceVertexBVH:
 def __init__(self,vertices,leaf_size=32):
  self.V=np.asarray(vertices,float);self.nodes=[];self.leaves=[]
  def build(ids):
   lo=self.V[ids].min(0);hi=self.V[ids].max(0);i=len(self.nodes);self.nodes.append(None)
   if len(ids)<=leaf_size:self.nodes[i]=dict(lo=lo,hi=hi,ids=ids,left=-1,right=-1);self.leaves.append(ids);return i
   axis=int(np.argmax(hi-lo));order=ids[np.argsort(self.V[ids,axis],kind='stable')];middle=len(order)//2;left=build(order[:middle]);right=build(order[middle:]);self.nodes[i]=dict(lo=lo,hi=hi,ids=None,left=left,right=right);return i
  self.root=build(np.arange(len(self.V)));cover=np.concatenate(self.leaves);assert len(cover)==len(self.V)and np.array_equal(np.sort(cover),np.arange(len(self.V)))
 def nearest_pair_certificate(self,points,saved_second_distance):
  P=np.asarray(points,float);radius=np.asarray(saved_second_distance,float);scale=max(1.,float(abs(P).max(initial=0)),float(abs(self.V).max(initial=0)));guard=128*np.finfo(float).eps*scale;outer=radius+guard;best=np.full((len(P),2),np.inf);stack=[(self.root,np.arange(len(P)))];steps=[];pairs=0
  while stack:
   i,ids=stack.pop();node=self.nodes[i];q=P[ids];offset=np.maximum(np.maximum(node['lo']-q,q-node['hi']),0);lower=np.sqrt(np.einsum('ni,ni->n',offset,offset,optimize=False));keep=lower<=outer[ids]+guard;retained=ids[keep];steps.append(dict(node=i,input_count=len(ids),retained_count=len(retained),pruned_count=int(np.count_nonzero(~keep)),node_lower=node['lo'],node_upper=node['hi']))
   if not len(retained):continue
   if node['ids']is None:stack.append((node['left'],retained));stack.append((node['right'],retained));continue
   delta=P[retained,None]-self.V[node['ids']][None];squared=np.einsum('pvi,pvi->pv',delta,delta,optimize=False);candidates=np.concatenate([best[retained],squared],axis=1);best[retained]=np.partition(candidates,1,axis=1)[:,:2];pairs+=squared.size
  return dict(first_two_distances=np.sqrt(best),numerical_search_guard_m=guard,all_points_have_two_candidates=bool(np.isfinite(best).all()),visited_nodes=steps,leaf_point_vertex_pairs=pairs,source_visual_vertices_NOT_native_mesh_vert=True,scope='All source visual vertices covered by leaves or conservative AABB pruning within saved second-radius. Guard is diagnostic, not any candidate source tolerance.')
