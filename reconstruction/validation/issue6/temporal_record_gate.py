"""Keep native temporal estimates with observed, physically usable camera rays."""

from issue6.geometry import camera, hand_translation


def rejection_reason(record, directly_observed, color_camera):
    # Bad calibration remains fatal. A bad individual native prediction does
    # not establish a usable hand observation and must remain unknown.
    camera(color_camera)
    if not directly_observed:
        return "not_directly_observed"
    try:
        hand_translation(record["camera"], color_camera)
    except (KeyError, TypeError, ValueError):
        return "invalid_virtual_camera"
    return None
