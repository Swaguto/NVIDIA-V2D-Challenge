"""Read only the owned instance; preserve the latest actual fallback receipt."""
from pathlib import Path
import datetime,json,subprocess,sys
S=Path(__file__).resolve().parents[1];out=Path(sys.argv[1]);assert not out.exists()
state=json.loads((S/'phase_state_20261018.json').read_text());fallback=Path(state['brev_current']['last_confirmed_provider_receipt']);prior=json.loads(fallback.read_text());assert prior['current_provider_status_verified']
start=datetime.datetime.now(datetime.timezone.utc).isoformat()
record=dict(schema='v2d.provider-status-attempt.v3',attempt_started_utc=start,instance='dngrtbldz',read_only=True,
 current_provider_status_verified=False,last_confirmed_provider_status=prior['provider_status'],last_confirmed_observed_utc=prior['checked_utc'],
 last_confirmed_receipt=str(fallback),VM_mutation_by_this_read=False,paid_work_queued=False)
def find(value):
 if isinstance(value,dict):
  if value.get('id')=='dngrtbldz':return value
  for v in value.values():
   r=find(v)
   if r is not None:return r
 elif isinstance(value,list):
  for v in value:
   r=find(v)
   if r is not None:return r
 return None
try:
 result=subprocess.run(['brev','ls','--json'],capture_output=True,text=True,timeout=20)
 record['CLI_exit_code']=result.returncode
 if result.returncode==0:
  row=find(json.loads(result.stdout));assert row is not None
  status=row.get('status')or row.get('state');assert isinstance(status,str)and status
  record.update(current_provider_status_verified=True,provider_status=status,checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 else:
  diagnostic=(result.stderr+result.stdout).lower();record['error']='CLI unauthorized'if 'unauthorized'in diagnostic or '401'in diagnostic else'CLI nonzero; provider status not inferred'
except subprocess.TimeoutExpired:record['error']='Read-only CLI timed out after20seconds; subprocess terminated'
except Exception as exc:record['error']='Read-only status parsing failed: '+type(exc).__name__
record['attempt_finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();out.write_text(json.dumps(record,indent=2)+'\n');out.chmod(0o600);print(json.dumps(record))
