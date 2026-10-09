"""Check fit evidence even when the worker enables optimized Python."""
import numpy as np


def verify_fit(initial, final, fixed_shape):
    with np.load(initial, allow_pickle=False) as initial_data, np.load(final, allow_pickle=False) as final_data:
        for key in ('trans', 'root_orient', 'pose_body', 'betas', 'cam_R', 'cam_t'):
            if not np.isfinite(final_data[key]).all():
                raise ValueError(f'Nonfinite final fit: {key}')
        if fixed_shape and not np.array_equal(initial_data['betas'], final_data['betas']):
            raise ValueError('Fixed shape changed during fit')
