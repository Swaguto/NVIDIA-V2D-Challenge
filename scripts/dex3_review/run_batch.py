"""Run selected checkpoint episodes through the explicit Docker Compose stages."""
from __future__ import annotations
import argparse,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--env-file',type=Path,required=True);p.add_argument('--episodes',default='11,40,41');g=p.add_mutually_exclusive_group();g.add_argument('--reuse-frozen-ik',dest='reuse_frozen_ik',action='store_true',default=True);g.add_argument('--recompute-ik',dest='reuse_frozen_ik',action='store_false');a=p.parse_args()
 episodes=[]
 for value in a.episodes.split(','):
  ep=int(value)
  if ep not in [2,11,12,13,18,21,23,24,31,40,41,42] or ep in episodes:raise ValueError('Unknown or repeated checkpoint episode')
  episodes.append(ep)
 compose=['docker','compose','--env-file',str(a.env_file.resolve()),'-f',str(ROOT/'docker/dex3-review/compose.yaml')]
 def command(service,*args):subprocess.run(compose+['run','--rm',service,*args],check=True)
 for ep in episodes:
  label=f'e{ep:03d}';bundle=f'/data/episodes/{label}/bundle';frozen=f'/data/episodes/{label}/retarget';out=f'/outputs/{label}'
  retarget=frozen if a.reuse_frozen_ik else out+'/retarget'
  if not a.reuse_frozen_ik:
   command('ik','retarget','--bundle',bundle,'--out',retarget,'--assets','/data/robot-assets')
   command('ik','compare','--reference',frozen,'--candidate',retarget)
  command('review','render','--bundle',bundle,'--retarget',retarget,'--out',out+'/sim','--assets','/data/robot-assets')
  command('review','package','--bundle',bundle,'--retarget',retarget,'--sim',out+'/sim','--human',f'/data/episodes/{label}/human-input.mp4','--out',out+'/review')
 print('BATCH_COMPLETE',len(episodes),'episodes; physical grasps remain unvalidated')
if __name__=='__main__':main()
