"""Explicit contact experiments; patch-only preserves native contact softness."""


def parameters(profile):
    if profile == "default":
        return {}
    if profile == "default_patch":
        return {"condim": 4}
    if profile not in {
        "firm_point",
        "firm_patch",
        "constraint_point",
        "constraint_patch",
    }:
        raise ValueError("Unknown explicit native hand contact profile")
    result = dict(
        priority=1,
        solimp=(0.99, 0.999, 0.001, 0.5, 2.0),
        condim=4 if profile.endswith("patch") else 3,
    )
    if profile.startswith("firm_"):
        result.update(ke=40000.0, kd=400.0)
    return result
