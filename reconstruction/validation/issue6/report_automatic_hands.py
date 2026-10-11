"""Summarize all frames without treating consistency as pose ground truth."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def intervals(rows, supported):
    result=[]; start=None
    dt=statistics.median(b['timestamp_s']-a['timestamp_s'] for a,b in zip(rows,rows[1:])) if len(rows)>1 else 0
    for i in range(len(rows)+1):
        missing=i<len(rows) and not supported[i]
        if missing and start is None:start=i
        if not missing and start is not None:
            result.append({'first_frame':rows[start]['frame'],'last_frame':rows[i-1]['frame'],
                           'frames':i-start,'duration_s':rows[i-1]['timestamp_s']-rows[start]['timestamp_s']+dt})
            start=None
    return result


def summarize(root):
    summary=json.loads((root/'quality/summary.json').read_text())
    rows=json.loads((root/'quality/frame-evidence.json').read_text())
    result={'episode':summary['episode'],'frames':len(rows),'quality_accepted':False,'hands':{}}
    for side in ('left','right'):
        items=[r['hands'].get(side) for r in rows]
        mask=[bool(item and item['accepted_by_consistency_proxies']) for item in items]
        gaps=intervals(rows,mask)
        result['hands'][side]={'proposals':sum(x is not None for x in items),
            'consistent_frames':sum(mask),'supported_frame_ids':[r['frame'] for r,ok in zip(rows,mask) if ok],
            'support_transitions':sum(a!=b for a,b in zip(mask,mask[1:])),
            'unsupported_intervals':gaps,'longest_unsupported_s':max((g['duration_s'] for g in gaps),default=0),
            'rejection_counts':dict(Counter(reason for item in items if item for reason in item['reasons']))}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode-roots',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    output={'scope':'Image/depth/motion consistency, not verified hand accuracy. Gaps include genuinely absent hands; support transitions are not a ground-truth flicker rate.',
            'episodes':[summarize(root) for root in a.episode_roots]}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
