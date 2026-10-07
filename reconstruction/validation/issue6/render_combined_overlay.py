"""Render observed CAD and final temporal MANO together in the calibrated camera."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import numpy as np
import pyrender
import torch
import trimesh
from PIL import Image, ImageDraw
from issue6.geometry import transform
from issue6.manifest import load, camera_from_object
from issue6.pipeline import load_mesh

parser = argparse.ArgumentParser()
parser.add_argument('--episode', type=int, required=True)
args = parser.parse_args()
label = f'e{args.episode:03d}'
root = Path(os.environ.get('V2D_ISSUE6_RUN_ROOT','/workspace/issue6-full-runs/calibrated-temporal')) / label
out = Path(os.environ.get('V2D_ISSUE6_REVIEW_ROOT','/workspace/pr32-evidence/review')) / label
out.mkdir(parents=True, exist_ok=True)
config = json.loads((root / 'align/job.json').read_text())
camera = config['camera']
frames = config['frames']
manifest = load(Path(os.environ.get('V2D_ISSUE6_INPUT_ROOT','/workspace/issue6-full-inputs')) / label / 'manifest.temporal.json')
source_frames = {f['id']: f for f in manifest['frames']}
meshes = {o['id']: load_mesh(o['mesh']) for o in manifest['objects']}
for frame in frames:
    frame['projected_objects'] = []
    for obj in manifest['objects']:
        pose = camera_from_object(source_frames[frame['id']], obj)
        if pose is not None:
            frame['projected_objects'].append({'id':obj['id'], 'camera_from_object':pose})
times = np.array([f['timestamp_s'] for f in frames])
assert np.allclose(np.diff(times), np.median(np.diff(times)), rtol=.01, atol=1e-6)
fps = float(1 / np.median(np.diff(times)))
spec = importlib.util.spec_from_file_location('combined_upstream', '/workspace/pr32/reconstruction/modules/v2d_hamer/lib/render_hands_aligned_video.py')
upstream = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upstream)
torch.set_num_threads(4)
mano = upstream._mano_layer('/workspace/mano')
records = upstream._frame_records(str(root / 'align/records'))
renderer = pyrender.OffscreenRenderer(camera['width'], camera['height'])
cam = pyrender.IntrinsicsCamera(camera['fx'], camera['fy'], camera['cx'], camera['cy'])
cv_to_gl = np.array([1, -1, -1])
object_colors = [(45, 220, 100), (240, 190, 50), (200, 100, 220)]
object_ids = sorted({o['id'] for f in frames for o in f['projected_objects']})
colors = {key: object_colors[i % len(object_colors)] for i, key in enumerate(object_ids)}
cache = {}
video = out / 'cad-temporal-hands.mp4'
process = subprocess.Popen(['ffmpeg', '-nostdin', '-y', '-v', 'error', '-f', 'rawvideo',
    '-pix_fmt', 'rgb24', '-s', f"{camera['width']}x{camera['height']}", '-r', str(fps),
    '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', str(video)], stdin=subprocess.PIPE)
selected = np.unique(np.linspace(0, len(frames)-1, 12).round().astype(int)).tolist()
montage = Image.new('RGB', (1920, 1712), (12, 17, 24))
counts = []
try:
    for index, frame in enumerate(frames):
        scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[.35, .35, .35])
        scene.add(cam, pose=np.eye(4))
        upstream._add_lights(scene, np.eye(4))
        hand_rows = records.get(frame['id'], [])
        for record in hand_rows:
            with torch.no_grad():
                vertices, faces = upstream._build_hand_mesh(record, mano)
            mesh = trimesh.Trimesh(vertices * cv_to_gl, faces, process=False)
            color = (235, 70, 70) if record['is_right'] else (60, 150, 235)
            scene.add(pyrender.Mesh.from_trimesh(mesh, material=upstream._material(color), smooth=False))
        for obj in frame['projected_objects']:
            mesh_source = meshes[obj['id']]
            vertices, faces = mesh_source.vertices, mesh_source.faces
            vertices = transform(vertices, obj['camera_from_object'])
            assert np.isfinite(vertices).all()
            mesh = trimesh.Trimesh(vertices * cv_to_gl, faces, process=False)
            scene.add(pyrender.Mesh.from_trimesh(mesh, material=upstream._material(colors[obj['id']]), smooth=False))
        rgb, depth = renderer.render(scene)
        source = np.array(Image.open(root / 'align/frames' / f"{frame['id']:06d}.png").convert('RGB'))
        mask = depth > 0
        source[mask] = (.45 * source[mask] + .55 * rgb[mask]).astype(np.uint8)
        image = Image.fromarray(source)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, camera['width'], 64), fill=(12, 17, 24))
        draw.text((12, 8), f"{label} | frame {frame['id']} | {frame['timestamp_s']:.3f}s | temporal hands + observed CAD", font=upstream._font(17), fill='white')
        draw.text((12, 36), 'Hands: left blue / right red  |  CAD: ' + ' / '.join(str(o) for o in object_ids), font=upstream._font(15), fill='white')
        if any(not r['diagnostics']['metric_valid'] for r in hand_rows):
            draw.text((12, camera['height']-28), 'Some hand depth unsupported: associated contacts remain unknown', font=upstream._font(15), fill=(255,220,80), stroke_width=2, stroke_fill=(10,10,10))
        process.stdin.write(np.asarray(image).tobytes())
        counts.append({'frame_id':frame['id'], 'timestamp_s':frame['timestamp_s'], 'hands':len(hand_rows), 'observed_objects':[o['id'] for o in frame['projected_objects']]})
        if index in selected:
            position = selected.index(index)
            x, y = position % 3 * 640, position // 3 * 428
            montage.paste(image.resize((640,400)), (x,y))
            ImageDraw.Draw(montage).text((x+8,y+405), f"{label} frame {frame['id']} / {frame['timestamp_s']:.3f}s", fill='white')
finally:
    process.stdin.close()
    renderer.delete()
assert process.wait() == 0
montage.save(out / 'cad-temporal-hands-montage.jpg', quality=94)
receipt = {'episode':args.episode, 'frames':counts, 'fps':fps, 'video':str(video),
    'sha256':hashlib.sha256(video.read_bytes()).hexdigest(), 'camera':camera,
    'basis':'Same observed CAD transforms/meshes as contact calculation and same final aligned temporal MANO records; joint depth-buffer rendering in source camera. Missing poses are omitted. No ground-truth poses or filled object poses.'}
(out / 'cad-temporal-hands-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'episode':args.episode,'frames':len(counts),'video':str(video),'sha256':receipt['sha256']}))
