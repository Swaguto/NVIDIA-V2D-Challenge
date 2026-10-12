"""Image-mask bounds and geometry-preserving native hand exports.

Bounding segmentation never removes rendered mesh pixels from evaluation.
Consistency-passing records remain unaccepted estimates, never display holds.
"""

import copy

import numpy as np


def clip_mask(mask, box):
    mask = np.asarray(mask, dtype=bool)
    box = np.asarray(box, dtype=float)
    if mask.ndim != 2 or box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError("Invalid current-frame detector box or image mask")
    height, width = mask.shape
    if not (box[0] < box[2] and box[1] < box[3]):
        raise ValueError("Reversed or empty detector box")
    x0, y0 = np.maximum(np.floor(box[:2]).astype(int), 0)
    x1, y1 = np.minimum(np.ceil(box[2:]).astype(int), [width, height])
    if not (x0 < x1 and y0 < y1):
        raise ValueError("Detector box has no image support")
    result = np.zeros_like(mask)
    result[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    return result


class DetectorBoundedPredictor:
    def __init__(self, predictor):
        self.predictor = predictor
        self.calls = []

    def set_image(self, image):
        self.predictor.set_image(image)

    def predict(self, *, box, multimask_output=False):
        masks, scores, logits = self.predictor.predict(
            box=box, multimask_output=multimask_output
        )
        constrained = np.stack([clip_mask(mask, box) for mask in masks])
        self.calls.append(
            {
                "box": np.asarray(box, float).tolist(),
                "unconstrained_pixels": [int(np.count_nonzero(mask)) for mask in masks],
                "constrained_pixels": [
                    int(np.count_nonzero(mask)) for mask in constrained
                ],
                "scores": np.asarray(scores, float).tolist(),
            }
        )
        return constrained, scores, logits


def export_native_record(source, offset, intrinsics, measurement, consistency, policy):
    """Keep the checked MANO scale, orientation and camera-ray translation."""
    if source.get("directly_observed") is not True:
        raise ValueError("Held or unsupported records cannot become observations")
    if (
        consistency.get("accepted_by_consistency_proxies") is not True
        or consistency.get("quality_accepted") is not False
    ):
        raise ValueError(
            "Export requires consistency evidence without accuracy acceptance"
        )
    scale = float(source.get("hand_scale", 1.0))
    translation = np.asarray(source["camera"]["pred_cam_t_full"], dtype=float)
    offset = np.asarray(offset, dtype=float)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid retained hand scale")
    if (
        translation.shape != (3,)
        or offset.shape != (3,)
        or not np.isfinite(translation).all()
        or not np.isfinite(offset).all()
    ):
        raise ValueError("Invalid measured hand translation")
    aligned = copy.deepcopy(source)
    aligned.update(
        cam_t=(translation + offset).tolist(),
        hand_scale=scale,
        intrinsics=copy.deepcopy(intrinsics),
        diagnostics={
            "metric_valid": True,
            "n_pixels": measurement["stereo_pixels"],
            "median_abs_residual_m": measurement["median_depth_residual_m"],
            "maximum_residual_m": policy["maximum_depth_residual_m"],
        },
        automatic_consistency=copy.deepcopy(consistency),
        quality_accepted=False,
        physical_task_accepted=False,
        geometry_export_verified=False,
    )
    return aligned
