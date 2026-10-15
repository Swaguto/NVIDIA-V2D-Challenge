import pathlib,tarfile,json,os,hashlib
r=pathlib.Path('/workspace/hand-continuity-repair');t=r/'all20-transfer';review=r/'review';c=review/'continuity_fix/all_episodes'
with tarfile.open(t/'final-code.tar.gz') as tar:tar.extractall(r/'code',filter='data')
# Link original completed videos, preserving their source experiments.
exec((t/'backup_review.py').read_text(), {})
for label in ['e041','e042']:
 current=c/label;sup=c/'superseded'/(label+'-forward');sup.parent.mkdir(parents=True,exist_ok=True)
 if current.exists() and not sup.exists():current.rename(sup)
 current.mkdir(parents=True,exist_ok=True)
 for p in (r/f'roi-recovery/{label}/refit-v1/review/{label}').iterdir():
  dest=current/p.name
  if dest.exists() and hashlib.sha256(dest.read_bytes()).digest()!=hashlib.sha256(p.read_bytes()).digest():
   old=c/'superseded'/(label+'-review-before-final')/p.name;old.parent.mkdir(parents=True,exist_ok=True)
   if not old.exists():os.link(dest,old)
   temp=dest.with_name(dest.name+'.new');os.link(p,temp);os.replace(temp,dest)
  elif not dest.exists():os.link(p,dest)
with tarfile.open(t/'review-sidecars.tar.gz') as tar:tar.extractall(review,filter='data')
expected=json.load(open(t/'delivery-hashes.json'));checked=0
for name,digest in expected.items():
 p=(review/'continuity_fix'/name).resolve();assert p.is_relative_to(review.resolve());assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,name;checked+=1
(review/'delivery-verification.json').write_text(json.dumps({'media_verified':checked,'sha256_matches_local':True},indent=2))
print('FINAL_PRIVATE_BACKUP_VERIFIED',checked,flush=True)
