"""Create a private, hash-bound replay input pack without model weights or credentials."""
from __future__ import annotations
import argparse,hashlib,json,tarfile
from pathlib import Path
EPISODES=[2,11,12,13,18,21,23,24,31,40,41,42]

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 root=a.root.resolve();files={};episodes=[];asset_lock=None
 def add(src,name):
  if src.is_symlink() or not src.is_file():raise ValueError('Expected regular input: '+str(src))
  files[name]=src
 for ep in EPISODES:
  label=f'e{ep:03d}';case=root/'episodes'/label
  for group in ['bundle','retarget']:
   for src in sorted((case/group).rglob('*')):
    if src.is_file():add(src,f'episodes/{label}/{group}/'+src.relative_to(case/group).as_posix())
  add(case/'review/human-input.mp4',f'episodes/{label}/human-input.mp4')
  contract=json.loads((case/'retarget/manifest.json').read_text())
  if asset_lock is not None and asset_lock!=contract['robot_assets']:raise ValueError('Mixed robot models')
  asset_lock=contract['robot_assets']
  for src in sorted((case/'review').glob('*.json')):add(src,f'expected/{label}/'+src.name)
  m=json.loads((case/'bundle/manifest.json').read_text());episodes.append({'episode':label,'frames':len(m['frames']),'bundle_manifest_sha256':hashlib.sha256((case/'bundle/manifest.json').read_bytes()).hexdigest()})
 assets=root/'retarget-code/robotic_grounding/source/robotic_grounding/robotic_grounding/assets'
 for rel,expected in asset_lock['files_sha256'].items():
  src=assets/rel
  if hashlib.sha256(src.read_bytes()).hexdigest()!=expected:raise ValueError('Robot asset mismatch')
  add(src,'robot-assets/'+rel)
 manifest={'schema':'v2d.dex3.private-review-pack.v1','episodes':episodes,'robot_assets':asset_lock,'files':{name:{'bytes':src.stat().st_size,'sha256':hashlib.sha256(src.read_bytes()).hexdigest()} for name,src in files.items()},'scope':'Private frozen repaired geometry, IK targets, human comparison videos and provided development CAD; not fresh reconstruction or final policy submission.'}
 # The pack carries no native MANO model .pkl, authentication material or learned weights.
 forbidden={'.pkl','.pth','.pt','.ckpt','.key','.pem'}
 if any(src.suffix.lower() in forbidden for src in files.values()):raise ValueError('Model or credential file rejected')
 a.out.parent.mkdir(parents=True,exist_ok=True);tmp=a.out.with_suffix(a.out.suffix+'.partial')
 if tmp.exists():raise FileExistsError(tmp)
 import io
 with tarfile.open(tmp,'w:gz',compresslevel=3) as t:
  for name,src in files.items():t.add(src,arcname='dex3-review/'+name,recursive=False)
  payload=json.dumps(manifest,indent=2).encode();info=tarfile.TarInfo('dex3-review/pack-manifest.json');info.size=len(payload);info.mode=0o600;t.addfile(info,io.BytesIO(payload))
 tmp.chmod(0o600);tmp.replace(a.out)
 receipt={'archive':a.out.name,'sha256':hashlib.sha256(a.out.read_bytes()).hexdigest(),'bytes':a.out.stat().st_size,'episodes':episodes,'members':len(files)+1,'schema':manifest['schema'],'private':True}
 a.out.with_suffix(a.out.suffix+'.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
if __name__=='__main__':main()
