"""Restore the separately licensed private runtime source snapshot after verification."""
import argparse,hashlib,json,shutil,tarfile
from pathlib import Path

def restore(archive,receipt,out):
 m=json.loads(receipt.read_text())
 if out.exists():raise FileExistsError('Use a fresh output directory')
 if hashlib.sha256(archive.read_bytes()).hexdigest()!=m['sha256']:raise ValueError('Runtime source archive hash mismatch')
 with tarfile.open(archive) as t:
  seen=set()
  for member in t.getmembers():
   p=Path(member.name)
   if not member.isfile() or p.is_absolute() or '..' in p.parts or member.name in seen:raise ValueError('Unsafe runtime member')
   seen.add(member.name)
   if member.name not in m['files'] or hashlib.sha256(t.extractfile(member).read()).hexdigest()!=m['files'][member.name]:raise ValueError('Changed runtime source')
  if seen!=set(m['files']):raise ValueError('Incomplete runtime source snapshot')
  out.mkdir(parents=True)
  for member in t.getmembers():
   p=out/member.name;p.parent.mkdir(parents=True,exist_ok=True)
   with t.extractfile(member) as src,p.open('xb') as f:shutil.copyfileobj(src,f)
 print('RUNTIME_SOURCES_VERIFIED',len(seen))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();restore(a.archive,a.receipt,a.out)
