from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import hmac,re
root=Path('/workspace/hand-continuity-repair/all20-transfer'); root.mkdir(exist_ok=True)
token=(root/'token').read_text().strip()
class Handler(BaseHTTPRequestHandler):
 def do_POST(self):
  if not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+token): self.send_error(403);return
  try:
   m=re.fullmatch(r'/chunk/(e\d{3})/(\d+)',self.path); assert m
   dest=root/(m[1]+'.tar.gz'); offset=int(m[2]); n=int(self.headers['Content-Length'])
   assert 0<n<=8*1024**2 and offset==(dest.stat().st_size if dest.exists() else 0) and offset+n<=2*1024**3
  except (ValueError,AssertionError,KeyError): self.send_error(409);return
  b=self.rfile.read(n)
  if len(b)!=n:self.send_error(400);return
  with dest.open('ab') as f:f.write(b)
  self.send_response(200);self.end_headers();self.wfile.write(b'OK')
 def log_message(self,*a):pass
HTTPServer(('0.0.0.0',19124),Handler).serve_forever()
