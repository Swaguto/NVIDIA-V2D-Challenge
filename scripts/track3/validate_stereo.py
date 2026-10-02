#!/usr/bin/env python3
"""Numerically validate ego stereo calibration (no image viewing required).

A patch search over a small +/-4px window is useless here: true horizontal disparity runs to
~63px (0.68m at a 7.5cm baseline), so such a search only ever finds noise. Instead:

1. Strict left-right consistency. StereoSGBM with disp12MaxDiff=1 keeps only matches whose
   disparity agrees when computed left-to-right and right-to-left. A correctly rectified pair
   yields few violations; deliberately shifting the right image by a few pixels must blow
   them up. That contrast is the evidence the calibration was applied correctly.

2. Planarity. A physical scene is dominated by a table/floor plane. If the depth solution is
   real, RANSAC finds a large, tight inlier set. Garbage depth gives scattered inliers.

3. Temporal stability: per-pixel depth std across a short window vs the depth level.
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/home/ubuntu")   # remote runtime location; module is on PYTHONPATH locally
from stereo_depth import load_rig, stereo_depth   # noqa: E402

VID = Path(os.environ.get("V2D_VIDEOS",
                    "/home/ubuntu/v2d_track3/data/hf/track_3/public/videos/chunk-000"))


def read_frame(path, idx):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, img = cap.read()
    cap.release()
    assert ok, f"could not read frame {idx} of {path}"
    return img


def lr_stats(l, r, nd=96, bs=5, tols=(0.5, 1.5, 3.0), tex_pct=60.0):
    """Left-right disparity disagreement, gated on texture, swept over tolerance.

    Matching (right, left) yields *negative* disparity, so the reverse matcher runs with
    minDisparity=-nd; a left pixel x with disparity d has its partner at x-d and must
    report the same magnitude. Textureless regions (blank table) are excluded because their
    disparities are meaningless; note 1px ~ 4% depth error at a 7.5cm baseline, so a strict
    threshold legitimately fires on a correct pair.
    """
    gl = cv2.cvtColor(l, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    common = dict(numDisparities=nd, blockSize=bs, P1=8 * 3 * bs, P2=32 * 3 * bs,
                  disp12MaxDiff=1, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    dl = cv2.StereoSGBM_create(minDisparity=0, **common).compute(gl, gr)
    dl = dl.astype(np.float32) / 16.0
    dr = cv2.StereoSGBM_create(minDisparity=-nd, **common).compute(gr, gl)
    dr = dr.astype(np.float32) / 16.0

    # local texture gate on the left image
    lap = cv2.Laplacian(gl, cv2.CV_32F, ksize=3)
    tex = np.abs(cv2.GaussianBlur(lap, (0, 0), 3))
    h, w = dl.shape
    # adaptive gate: keep the most-textured `tex_pct`% of pixels. An absolute threshold on
    # |Laplacian| is meaningless without knowing the image's contrast scale.
    thr = np.percentile(tex, tex_pct)
    ys, xs = np.nonzero((dl > 0) & (tex >= thr))
    ys, xs = ys[::2], xs[::2]
    if len(xs) == 0:
        return {t: float("nan") for t in tols}, 0
    d = dl[ys, xs]
    xr = np.round(xs - d).astype(np.int32)
    ok = (xr >= 0) & (xr < w)
    xs, xr, d, ys = xs[ok], xr[ok], d[ok], ys[ok]
    dk = dr[ys, xr]
    good = dk < 0
    if good.sum() < 100:
        return {t: float("nan") for t in tols}, 0
    diff = np.abs(np.abs(dk[good]) - d[good])
    return {t: float((diff > t).mean()) for t in tols}, int(good.sum())


def plane_inliers(depth, valid, P):
    """RANSAC plane fit on the reconstructed cloud; returns inlier ratio and RMS error."""
    ys, xs = np.nonzero(valid & (depth > 0.2) & (depth < 4.0))
    z = depth[ys, xs].astype(np.float64)
    x = (xs - P[0, 2]) * z / P[0, 0]
    y = (ys - P[1, 2]) * z / P[1, 1]
    pts = np.stack([x, y, z], 1).astype(np.float32)
    if len(pts) < 500:
        return 0.0, float("nan"), 0
    if len(pts) > 60000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 60000, replace=False)]
    # RANSAC over random triples, keeping near-horizontal support surfaces
    rng = np.random.default_rng(0)
    n = pts.shape[0]
    best = None
    for _ in range(200):
        i = rng.choice(n, 3, replace=False)
        p0, p1, p2 = pts[i]
        nv = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(nv)
        if nn < 1e-9:
            continue
        nv /= nn
        if nv[2] < 0:
            nv = -nv
        if abs(nv[2]) < 0.5:      # want a near-horizontal support surface
            continue
        d = np.abs((pts - p0) @ nv)
        k = d < 0.01
        c = int(k.sum())
        if best is None or c > best[0]:
            best = (c, np.sqrt((d[k] ** 2).mean()) if k.any() else np.nan)
    if best is None:
        return 0.0, float("nan"), 0
    return best[0] / n, float(best[1]), int(n)


def main():
    ep = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    frame = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    rig = load_rig()
    vl = VID / f"observation.images.{rig.left}" / f"episode_{ep:06d}.mp4"
    vr = VID / f"observation.images.{rig.right}" / f"episode_{ep:06d}.mp4"
    il, ir = read_frame(vl, frame), read_frame(vr, frame)
    pl, pr = rig.rectify_pair(il, ir)
    f, b = rig.fx_rect, abs(rig.P2[0, 3]) / rig.P1[0, 0]
    print(f"ep {ep} frame {frame}  fx_rect={f:.3f}  baseline={b:.6f} m")

    print("\n=== 1. left-right consistency (deliberate mis-rectification as control) ===")
    base, n = lr_stats(pl, pr)
    print(f"  rectified as published : " + "  ".join(f">{t}px {base[t]:.1%}" for t in base)
          + f"   ({n} textured px)")
    for sh in (2, 4):
        v, _ = lr_stats(pl, np.roll(pr, sh, axis=0))
        print(f"  right image shifted {sh}px: " + "  ".join(f">{t}px {v[t]:.1%}" for t in v))

    print("\n=== 2. support-surface planarity (RANSAC) ===")
    for f_ in (frame, frame + 40):
        d, v = stereo_depth(rig, read_frame(vl, f_), read_frame(vr, f_))
        ratio, rms, n = plane_inliers(d, v, rig.P1)
        print(f"  frame {f_:>4}: inliers {ratio:.1%} of {n} pts, RMS {rms * 1000:.1f} mm")

    print("\n=== 3. temporal depth stability ===")
    st = [stereo_depth(rig, read_frame(vl, f_), read_frame(vr, f_))[0]
          for f_ in range(frame, frame + 7)]
    st = np.stack(st)[:, 250:550, 350:900]
    m = (st > 0.15) & (st < 4.0)
    allv = m.all(axis=0)
    mean = st[:, allv].mean(axis=0)
    std = st[:, allv].std(axis=0)
    print(f"  depth level {mean.mean():.3f} m  temporal std median "
          f"{np.median(std) * 1000:.1f} mm ({np.median(std) / mean.mean() * 100:.2f}%)")

    # Decision: correctly-rectified must beat deliberate mis-rectification at every tolerance.
    ctrl = [lr_stats(pl, np.roll(pr, s, axis=0))[0] for s in (2, 4)]
    ok_lr = all(base[t] < c[t] for c in ctrl for t in base)
    print(f"\nVERDICT lr_consistency={'PASS' if ok_lr else 'FAIL'} "
          f"(rectified beats both controls at every tolerance)")


if __name__ == "__main__":
    main()