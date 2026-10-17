"""Apply explicit data/support hooks to the pinned upstream runtime; fail on drift."""

import argparse
import hashlib
import json
from pathlib import Path

PATCHES = {
    "runtime/sim.py": (
        "4ede920bd4b07a231b31ab9e122029977fe46217797944c900f0bc60582dcf6f",
        "    return newton.solvers.SolverMuJoCo(\n",
        "    from v2d_dex3_flashchord.sim import NativeContactSimConfig, build_solver as build_native\n"
        "    if isinstance(cfg, NativeContactSimConfig):\n"
        "        return build_native(model, cfg)\n"
        "    return newton.solvers.SolverMuJoCo(\n",
    ),
    "scene/support.py": (
        "8d82f6008f2310667bef6862dfa3f453d3bdd188c79c7283a26e288a5a19e1c2",
        "    if not path.is_file():\n",
        "    from v2d_dex3_flashchord.support import has_mesh_support, add_mesh_support\n"
        "    if path.is_file() and has_mesh_support(path):\n"
        "        return add_mesh_support(builder, path)\n"
        "    if not path.is_file():\n",
    ),
    "data/loader.py": (
        "7af2e2a6791f540184040da43a6962adffe53e777aa3bd1674cf15936767b8c3",
        '    if schema_version == "motion_v1":',
        '    if schema_version == "dex3_dual_hand_v1":\n'
        "        from v2d_dex3_flashchord.reference import load_dex3\n"
        "        return load_dex3(row.path, control_fps=control_fps, motion_speed=motion_speed,\n"
        "                         source_frame_playback=source_frame_playback)\n"
        '    if schema_version == "motion_v1":',
    ),
    "assets/registry.py": (
        "5bf22299ec1bac0a3c99cc26b4ea1f7dd16f508e4d1b1786d586b22fe3a30078",
        "    metadata = reference.metadata\n    path = support_usda_for_parquet",
        "    metadata = reference.metadata\n"
        '    if metadata.schema_version == "dex3_dual_hand_v1":\n'
        '        support = Path(metadata.source_path).parent / "support.usda"\n'
        "        return support if support.is_file() else None\n"
        "    path = support_usda_for_parquet",
    ),
}


def install(runtime, extend_existing=False):
    receipt = {}
    replacements = []
    existing_path = runtime / "dex3-extension-patches.json"
    existing = json.loads(existing_path.read_text()) if existing_path.exists() else {}
    for relative, (expected, old, new) in PATCHES.items():
        path = runtime / "src/flash_chord" / relative
        source = path.read_bytes()
        current = hashlib.sha256(source).hexdigest()
        prior = existing.get(relative, {})
        if (
            extend_existing
            and prior.get("before_sha256") == expected
            and current == prior.get("after_sha256")
        ):
            receipt[relative] = prior
            continue
        if (
            hashlib.sha256(source).hexdigest() != expected
            or source.decode().count(old) != 1
        ):
            raise ValueError(f"Upstream source drift: {path}")
        changed = source.decode().replace(old, new)
        replacements.append((path, changed))
        receipt[relative] = dict(
            before_sha256=expected,
            after_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        )
    for path, changed in replacements:
        path.write_text(changed)
    (runtime / "dex3-extension-patches.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", required=True, type=Path)
    p.add_argument("--extend-existing", action="store_true")
    args = p.parse_args()
    print(json.dumps(install(args.runtime, args.extend_existing), indent=2))
