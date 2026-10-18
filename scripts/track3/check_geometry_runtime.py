"""Require populated metric TSDF geometry before running episode reconstruction."""

import json
from pathlib import Path

import numpy as np
import open3d as o3d


def check():
    colour = np.full((100, 100, 3), 128, np.uint8)
    depth = np.full((100, 100), 0.5, np.float32)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(colour),
        o3d.geometry.Image(depth),
        depth_scale=1.0,
        depth_trunc=2.5,
        convert_rgb_to_intensity=False,
    )
    np.testing.assert_array_equal(np.asarray(rgbd.depth), depth)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(100, 100, 100.0, 100.0, 50.0, 50.0)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.003,
        sdf_trunc=0.018,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    volume.integrate(rgbd, intrinsic, np.eye(4))
    mesh = volume.extract_triangle_mesh()
    vertices = np.asarray(mesh.vertices)
    if (
        len(vertices) < 100
        or len(mesh.triangles) < 100
        or not np.isfinite(vertices).all()
    ):
        raise RuntimeError(
            "Metric TSDF integration is broken; refusing episode reconstruction"
        )
    if abs(np.median(vertices[:, 2]) - 0.5) > 0.006:
        raise RuntimeError("Metric TSDF surface changed depth scale")
    return dict(
        schema="v2d.metric-tsdf-runtime.v1",
        open3d=o3d.__version__,
        vertices=len(vertices),
        triangles=len(mesh.triangles),
        expected_depth_m=0.5,
        median_depth_m=float(np.median(vertices[:, 2])),
        passed=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = check()
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
