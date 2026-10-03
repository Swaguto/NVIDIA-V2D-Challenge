"""Quantitative proxy for what the overlay video shows: how well the CAD silhouette at the
estimated pose agrees with the observed stereo depth inside the object region.

Reports, per frame:
  fill      fraction of the CAD silhouette that lands on valid depth (in-frame-ness)
  z_match   fraction of CAD points whose predicted depth agrees with median observed depth
  aspect    silhouette height/width, a shape sanity check
"""
import sys, json
from pathlib import Path
import numpy as np, trimesh, cv2

sys.path.insert(0, "/home/ubuntu")
from stereo_depth import load_rig, stereo_depth

rig = load_rig()
EP = 18
ep = f"e{EP:03d}"
OBJ = "mini_dust_pan"
PRED = Path(sys.argv[1] if len(sys.argv) > 1 else f"/home/ubuntu/pred_e18_const")
MESH = f"/home/ubuntu/v2d_track3/data/hf/track_3/public/mesh/{OBJ}/{OBJ}.glb"
V = np.asarray(trimesh.load(MESH, force="mesh", process=False).vertices, np.float64)
base = PRED / ep / "ego_object_poses" / ep / OBJ / "object_to_cam" / "ego_cam_c"

vid = Path("/home/ubuntu/v2d_track3/data/hf/track_3/public/videos/chunk-000")
cap = cv2.VideoCapture(str(vid / f"observation.images.{rig.left}" / f"episode_{EP:06d}.mp4"))
capr = cv2.VideoCapture(str(vid / f"observation.images.{rig.right}" / f"episode_{EP:06d}.mp4"))
want = {int(p.stem) for p in sorted(base.glob("*.json"))[::40]}
rows = []; i = 0
while want:
    ok, L = cap.read(); okr, Rr = capr.read()
    if not ok: break
    if i in want:
        d, _ = stereo_depth(rig, L, Rr)
        r = json.loads((base / f"{i:06d}.json").read_text())
        t = np.array(r["translation"]); q = np.array(r["rotation"]); q /= np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                      [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        X = (R @ V.T).T + t; zz = X[:, 2]; okz = zz > 1e-3
        u = (rig.P1[0,0]*X[:,0]/zz + rig.P1[0,2]).astype(int)
        v = (rig.P1[1,1]*X[:,1]/zz + rig.P1[1,2]).astype(int)
        inb = okz & (u >= 0) & (u < 1280) & (v >= 0) & (v < 800)
        fill = inb.sum() / max(okz.sum(), 1)
        agree = np.zeros(len(V), bool)
        dv = d[v[inb], u[inb]]
        good = dv > 0
        if good.sum() > 10:
            med = np.median(dv[good])
            agree[inb] = np.abs(dv - med) < 0.05
        rows.append((i, fill, float(agree[inb].mean()) if inb.sum() else 0.0,
                     float(np.median(zz[okz])), int(np.ptp(v[inb])), int(np.ptp(u[inb]))))
        want.discard(i)
    i += 1
print(f"{'frame':>6} {'on-screen':>10} {'z_agree':>8} {'z_pred':>7} {'h_px':>5} {'w_px':>5}")
for r_ in rows:
    print(f"{r_[0]:6d} {r_[1]*100:9.1f}% {r_[2]*100:7.1f}% {r_[3]:7.2f} {r_[4]:5d} {r_[5]:5d}")
if rows:
    print(f"\nmean on-screen={np.mean([r[1] for r in rows])*100:.1f}%  "
          f"mean z_agree={np.mean([r[2] for r in rows])*100:.1f}%")
