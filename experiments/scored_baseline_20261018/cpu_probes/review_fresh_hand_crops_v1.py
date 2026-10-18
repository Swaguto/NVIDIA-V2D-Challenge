"""Source-only review of the prepared inference crops; no models or rendering."""
from pathlib import Path
import json,hashlib
from PIL import Image,ImageDraw
S=Path(__file__).resolve().parents[1];P=S/'e037-fresh-source-hand-request-v1';request=json.load(open(P/'request.json'));O=S/'e037-fresh-source-crop-review-v1';O.mkdir();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();panels=[];rows=[]
for row in request['frames']:
 image=Image.open(row['image']).convert('RGB');c=row['crop'];box=(round(c['cx']-c['size']/2),round(c['cy']-c['size']/2),round(c['cx']+c['size']/2),round(c['cy']+c['size']/2));patch=image.crop(box).resize((320,320));panel=Image.new('RGB',(320,350),'black');panel.paste(patch,(0,30));draw=ImageDraw.Draw(panel);draw.text((8,7),f"frame {row['frame_idx']}: inferred source crop",fill='white');panel.save(O/f"{row['frame_idx']:06d}-crop.jpg");panels.append(panel);rows.append(dict(frame_idx=row['frame_idx'],source_image_sha256=sha(Path(row['image'])),crop_box=list(box),outside_source_pixels=bool(box[0]<0 or box[1]<0 or box[2]>image.width or box[3]>image.height)))
grid=Image.new('RGB',(1280,700));
for i,panel in enumerate(panels):grid.paste(panel,((i%4)*320,(i//4)*350))
grid.save(O/'source-crop-grid.jpg');r=dict(schema='v2d.source-hand-crop-review.v1',producer_sha256=sha(Path(__file__)),request_sha256=sha(P/'request.json'),rows=rows,not_geometry_or_pose_validation=True,new_model_inference=False);(O/'report.json').write_text(json.dumps(r,indent=2)+'\n');print('Prepared',len(rows),'source crop previews')
