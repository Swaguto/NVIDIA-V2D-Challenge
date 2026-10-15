"""Carry wearer image landmarks across detector gaps using checked image motion.

Seeds must be directly observed and lie on the saved hand segmentation. Each
step requires forward/backward agreement in the actual source images. These
are inferred image tracks, never new direct observations or contact evidence.
"""

import cv2
import numpy as np


def propagate_landmarks(images, masks, observations, times, max_gap_seconds=1.5):
    result = np.zeros((len(images), 21, 3), float)
    previous = None
    last_anchor = None
    points = None
    active = np.zeros(21, bool)
    for i, (image, mask, observed) in enumerate(zip(images, masks, observations)):
        gray = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"Cannot read source image: {image}")
        h, w = gray.shape
        if observed is not None:
            kp = np.asarray(observed, float)
            result[i] = kp
            points = kp[:, :2].astype(np.float32)
            semantic = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE) if mask else None
            active = (
                (kp[:, 2] > 0)
                & (points[:, 0] >= 2)
                & (points[:, 0] < w - 2)
                & (points[:, 1] >= 2)
                & (points[:, 1] < h - 2)
            )
            if semantic is None:
                active[:] = False
            else:
                semantic = cv2.dilate((semantic > 0).astype(np.uint8), np.ones((7, 7), np.uint8))
                for k in np.flatnonzero(active):
                    x, y = np.rint(points[k]).astype(int)
                    active[k] = bool(semantic[y, x])
            last_anchor = times[i]
        elif (
            points is not None
            and previous is not None
            and times[i] - last_anchor <= max_gap_seconds
            and active.sum() >= 4
        ):
            ids = np.flatnonzero(active)
            before = points[ids, None, :]
            after, sf, _ = cv2.calcOpticalFlowPyrLK(previous, gray, before, None, winSize=(21, 21), maxLevel=3)
            back, sb, _ = cv2.calcOpticalFlowPyrLK(gray, previous, after, None, winSize=(21, 21), maxLevel=3)
            candidate = after[:, 0]
            good = (
                (sf[:, 0] == 1)
                & (sb[:, 0] == 1)
                & (np.linalg.norm(back[:, 0] - before[:, 0], axis=1) <= 1.5)
                & np.isfinite(candidate).all(axis=1)
                & (candidate[:, 0] >= 2)
                & (candidate[:, 0] < w - 2)
                & (candidate[:, 1] >= 2)
                & (candidate[:, 1] < h - 2)
            )
            active[ids[~good]] = False
            points[ids[good]] = candidate[good]
            if active.sum() >= 4:
                result[i, active, :2] = points[active]
                result[i, active, 2] = 0.5
        else:
            active[:] = False
        previous = gray
    return result
