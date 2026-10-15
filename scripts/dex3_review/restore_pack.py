"""Verify a private archive before extracting it to a fresh data directory."""
from __future__ import annotations
import argparse,hashlib,json,tarfile
from pathlib import Path

def restore(archive,expected,out):
 if out.exists():raise FileExistsError('Use a fresh directory: '+str(out))
 h=hashlib.sha256()
 with archive.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 if h.hexdigest()!=expected:raise ValueError('Archive SHA256 mismatch')
 with tarfile.open(archive) as t:
  members=t.getmembers();seen=set();total=0
  if len(members)>50000:raise ValueError('Too many archive members')
  for m in members:
   p=Path(m.name)
   if p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0]!='dex3-review' or not m.isfile():raise ValueError('Unsafe member: '+m.name)
   if m.name in seen:raise ValueError('Duplicate archive member')
   seen.add(m.name);total+=m.size
   if total>20*1024**3:raise ValueError('Archive exceeds the supported size')
  manifest=json.load(t.extractfile('dex3-review/pack-manifest.json'))
  expected_names={'dex3-review/'+x for x in manifest['files']}|{'dex3-review/pack-manifest.json'}
  if seen!=expected_names:raise ValueError('Archive inventory differs from its manifest')
  # Hash all regular members before writing anything to disk.
  for name,record in manifest['files'].items():
   f=t.extractfile('dex3-review/'+name);h=hashlib.sha256();size=0
   for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk);size+=len(chunk)
   if h.hexdigest()!=record['sha256'] or size!=record['bytes']:raise ValueError('Changed input: '+name)
  out.mkdir(parents=True)
  for m in members:
   dst=out/Path(m.name).relative_to('dex3-review');dst.parent.mkdir(parents=True,exist_ok=True)
   with t.extractfile(m) as src,dst.open('xb') as f:
    import shutil;shutil.copyfileobj(src,f)
   dst.chmod(0o600)
 print('PRIVATE_PACK_VERIFIED',len(manifest['files']),'files',len(manifest['episodes']),'episodes')
 return manifest
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();restore(a.archive,a.sha256,a.out)
