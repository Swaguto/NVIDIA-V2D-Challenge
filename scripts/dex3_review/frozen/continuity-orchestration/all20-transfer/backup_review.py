import pathlib,json,os
r=pathlib.Path('/workspace/hand-continuity-repair');t=r/'all20-transfer';review=r/'review';c=review/'continuity_fix';states={}
for n in ['status','retry-status','parallel-status','worker-c-status','worker-d-status']:
 p=t/(n+'.json')
 if p.exists():states.update(json.loads(p.read_text()))
def link(source,target):
 target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists():return
 os.link(source,target)
for label,row in states.items():
 if not (t/(label+'-review.tar')).exists():continue
 out=pathlib.Path(row['output']);files={'continuity-receipt.json':out/'runs'/label/'continuity-receipt.json','cad-temporal-hands.mp4':out/'videos'/label/'cad-temporal-hands.mp4','hand-tracking.mp4':out/'videos'/label/'hand-tracking.mp4','baseline-cad-temporal-hands.mp4':out/'baseline-videos'/label/'cad-temporal-hands.mp4'}
 for name,p in files.items():link(p,c/'all_episodes'/label/name)
for label in ['e012','e040']:
 for name in ['cad-temporal-hands.mp4','hand-tracking.mp4']:link(r/'final-videos'/label/name,c/(label+'-'+name))
 source=pathlib.Path('/workspace/episode-review-20261014/review-artifacts')/(label+'-cad-temporal-hands.mp4')
 if source.is_file():link(source,review/source.name)
print('PRIVATE_REVIEW_MEDIA_BACKUP',sum(1 for p in review.rglob('*.mp4')))
