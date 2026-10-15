"""Restore only hash-locked Dex3 URDF/mesh files into a disposable container."""
import argparse,hashlib,json,shutil
from pathlib import Path

def restore(source,destination):
 source=source.resolve();destination=destination.resolve()
 # Exported pack stores the single model lock at its root.
 manifest=json.loads((source.parent/'pack-manifest.json').read_text())
 lock=manifest['robot_assets']['files_sha256']
 checked=[]
 for rel,digest in lock.items():
  p=Path(rel)
  if p.is_absolute() or '..' in p.parts or p.parts[0] not in ['meshes','urdfs']:raise ValueError('Unsafe asset path')
  src=(source/p).resolve()
  if not src.is_relative_to(source) or not src.is_file():raise ValueError('Missing or escaping asset')
  if hashlib.sha256(src.read_bytes()).hexdigest()!=digest:raise ValueError('Changed asset: '+rel)
  checked.append((src,destination/p))
 for src,dst in checked:
  dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
 print('ROBOT_ASSETS_VERIFIED',len(checked))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);a=p.parse_args();restore(a.source,a.destination)
