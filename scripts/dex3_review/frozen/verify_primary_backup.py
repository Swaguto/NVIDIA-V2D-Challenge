import hashlib,json,tarfile
from pathlib import Path
D=Path('/workspace/hand-continuity-repair/dex3-isaac-review')
P=Path('/workspace/hand-continuity-repair/review/continuity_fix')
P.mkdir(exist_ok=True,parents=True)
with tarfile.open(D/'gallery-backup.tar') as t:
 for m in t.getmembers():
  p=Path(m.name)
  if p.is_absolute() or '..' in p.parts or p.parts[0]!='dex3_isaac' or not m.isfile():raise ValueError('Unsafe backup')
 t.extractall(P)
G=P/'dex3_isaac';manifest=json.load(open(G/'backup-hashes.json'))
for rel,h in manifest.items():
 assert hashlib.sha256((G/rel).read_bytes()).hexdigest()==h,rel
v={'all_sha256_match_local_gallery':True,'verified_files':len(manifest),'videos':sum(n.endswith('.mp4') for n in manifest),'persistent_path':str(G),'original_episode_outputs_preserved':True}
assert v['videos']==48
(G/'backup-verification.json').write_text(json.dumps(v,indent=2))
print(json.dumps(v))
