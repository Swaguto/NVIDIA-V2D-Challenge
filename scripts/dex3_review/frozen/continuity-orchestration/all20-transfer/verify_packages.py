from pathlib import Path
import json,hashlib,concurrent.futures
root=Path('/workspace/hand-continuity-repair/all20-transfer');rows={}
for file in root.glob('hand-all20-*/audit.json'):
 for row in json.loads(file.read_text()):
  if not row['missing']:rows[row['episode']]=row
assert len(rows)==17

def verify(item):
 label,row=item;p=root/(label+'.tar.gz');h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8*1024**2):h.update(b)
 assert h.hexdigest()==row['sha256'] and p.stat().st_size==row['bytes'],label
 return {'episode':label,'sha256':h.hexdigest(),'bytes':p.stat().st_size,'verified':True}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(verify,sorted(rows.items())))
(root/'input-package-verification.json').write_text(json.dumps(results,indent=2));print('VERIFIED_INPUT_PACKAGES',len(results),sum(r['bytes'] for r in results),flush=True)
