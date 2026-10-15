"""List native-cache symlinks that need explicit preservation during migration."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path

def audit(root):
 root=root.resolve();rows=[];count=0
 for name in ['inputs','runtime-additions','roi-recovery']:
  start=root/name
  if not start.exists():continue
  for current,dirs,files in os.walk(start,followlinks=False):
   for name in dirs+files:
    p=Path(current)/name
    if p.is_symlink():
     count+=1;target=p.resolve();exists=target.exists()
     if not target.is_relative_to(root) or not exists:
      rows.append({'path':str(p.relative_to(root)),'target':str(target),'exists':exists,'external':not target.is_relative_to(root)})
 return {'scope':'Native inputs, runtime-additions and ROI recovery; installed virtual environments are excluded','workspace':str(root),'symlinks_checked':count,'external_or_broken_links':rows,'no_external_or_broken_links_in_scanned_roots':not rows,'not_a_complete_input_audit':True}
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path);a=p.parse_args();r=audit(a.root)
 if a.out:a.out.write_text(json.dumps(r,indent=2))
 print(json.dumps({'symlinks_checked':r['symlinks_checked'],'external_or_broken_links':len(r['external_or_broken_links']),'no_external_or_broken_links_in_scanned_roots':r['no_external_or_broken_links_in_scanned_roots']}))
