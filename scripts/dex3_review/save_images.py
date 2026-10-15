"""Save the three locally built images and record exact image IDs and archive hash."""
from __future__ import annotations
import argparse,hashlib,json,subprocess,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--tag',default='checkpoint');a=p.parse_args()
 if a.out.exists() or a.out.with_suffix(a.out.suffix+'.partial').exists():raise FileExistsError('Use a fresh image archive path')
 names=[n+':'+a.tag for n in ['v2d-dex3-ik','v2d-dex3-review','v2d-hand-continuity']]
 metadata=json.loads(subprocess.check_output(['docker','image','inspect',*names],text=True))
 for row in metadata:
  if row['Architecture']!='amd64' or row['Os']!='linux':raise ValueError('Expected Linux amd64 image')
 temp=a.out.with_suffix(a.out.suffix+'.partial');a.out.parent.mkdir(parents=True,exist_ok=True)
 # Compressed Docker archive, with each pipeline return code checked.
 with temp.open('xb') as out:
  save=subprocess.Popen(['docker','image','save',*names],stdout=subprocess.PIPE)
  zip_process=subprocess.Popen(['zstd','-T2','-1','-c'],stdin=save.stdout,stdout=out)
  save.stdout.close();zip_rc=zip_process.wait();save_rc=save.wait()
 if zip_rc or save_rc:raise RuntimeError('Image export failed; preserved partial archive')
 temp.chmod(0o600);temp.replace(a.out);h=hashlib.sha256()
 with a.out.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 receipt={'archive':a.out.name,'bytes':a.out.stat().st_size,'sha256':h.hexdigest(),'compression':'zstd','images':[{'tags':r['RepoTags'],'id':r['Id'],'os':r['Os'],'architecture':r['Architecture'],'revision':r['Config'].get('Labels',{}).get('org.opencontainers.image.revision')} for r in metadata],'scope':'Local image build/export; no GPU container rendering or Brev acceptance implied.'}
 a.out.with_suffix(a.out.suffix+'.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
