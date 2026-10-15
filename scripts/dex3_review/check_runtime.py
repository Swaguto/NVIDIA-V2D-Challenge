"""Check every recorded runtime distribution without launching the simulator."""
import argparse,importlib.metadata,json
p=argparse.ArgumentParser();p.add_argument('--runtime',choices=['ik','continuity'],required=True);p.add_argument('--lock',type=__import__('pathlib').Path,required=True);a=p.parse_args()
mismatches=[];count=0
for line in a.lock.read_text().splitlines():
 if not line.strip() or line.startswith('#'):continue
 name,expected=line.split('==');count+=1
 try:actual=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:actual=None
 if actual!=expected:mismatches.append({'name':name,'expected':expected,'actual':actual})
if a.runtime=='ik':
 import pinocchio,pink,viser,torch
else:
 import numpy,scipy,torch,trimesh,pyrender
print(json.dumps({'runtime':a.runtime,'checked_packages':count,'mismatches':mismatches},indent=2))
if mismatches:raise SystemExit(1)
