"""Lesson 4.10 - why one pixel is a ray, and what that costs you in millimetres.

Pure numpy. No robot, no camera, no LeRobot. Runs anywhere, and every number
quoted in the lesson comes out of this file.

Run:  python pixel_ray_ambiguity.py
      python pixel_ray_ambiguity.py --hfov 78 --width 1280
"""
import argparse
import numpy as np

# A plain 640x480 webcam. Horizontal field of view is the one spec that is
# usually on the box; everything else here follows from it.
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_HFOV_DEG = 60.0

# Where an SO-101 actually works: the table is roughly a third of a metre from
# a camera clamped at the back of the desk.
WORKING_DEPTH_M = 0.35


def focal_length_px(width_px, hfov_deg):
    """Pinhole focal length in pixels, from image width and horizontal FOV.

    Half the sensor width subtends half the field of view, so
    (width / 2) = f * tan(hfov / 2).
    """
    return (width_px / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)


def project(point_cam, f_px, principal_point):
    """3D point in CAMERA coordinates (metres) -> pixel (u, v).

    This is the whole pinhole model. Note the division by z: that division is
    the step that throws depth away.
    """
    x, y, z = point_cam
    if z <= 0:
        raise ValueError(f"point is behind the lens (z={z}); nothing to project")
    cx, cy = principal_point
    return (f_px * x / z + cx, f_px * y / z + cy)


def mm_per_pixel(depth_m, f_px):
    """How much real-world sideways distance one pixel covers, at this depth."""
    return 1000.0 * depth_m / f_px


def apparent_width_px(object_width_m, depth_m, f_px):
    """How many pixels wide an object of a known size looks, at this depth."""
    return f_px * object_width_m / depth_m


def lateral_error_from_depth_error(true_x_m, true_z_m, assumed_z_m):
    """Classical pipeline: back-project a pixel using the WRONG depth.

    x is recovered as (u - cx) * z / f, so the recovered x scales exactly with
    the depth you assumed. Returns (recovered_x, signed error) in metres.
    """
    recovered = true_x_m * assumed_z_m / true_z_m
    return recovered, recovered - true_x_m


def pixel_shift_from_camera_shift(camera_shift_m, depth_m, f_px):
    """Move the camera sideways by this much; everything in frame moves by
    this many pixels the other way."""
    return f_px * camera_shift_m / depth_m


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--hfov", type=float, default=DEFAULT_HFOV_DEG)
    parser.add_argument("--depth", type=float, default=WORKING_DEPTH_M)
    args = parser.parse_args()

    f = focal_length_px(args.width, args.hfov)
    pp = (args.width / 2.0, args.height / 2.0)
    z = args.depth

    print(f"{args.width}x{args.height} camera, {args.hfov:.0f} deg horizontal FOV")
    print(f"focal length            {f:.1f} px")
    print(f"one pixel at {z * 100:.0f} cm      {mm_per_pixel(z, f):.2f} mm")
    print()

    print("1. The ray. Three points, one pixel.")
    for k in (0.5, 1.0, 2.0):
        p = (0.06 * k, 0.02 * k, z * k)
        u, v = project(p, f, pp)
        print(f"   point {p[0]:+.3f}, {p[1]:+.3f}, {p[2]:.3f} m  ->  pixel ({u:.1f}, {v:.1f})")
    print("   Scale a point along the ray and the pixel does not move at all.")
    print()

    print("2. Scale ambiguity. Two different objects, identical footprint.")
    for width_m, depth_m in ((0.04, 0.35), (0.08, 0.70)):
        w = apparent_width_px(width_m, depth_m, f)
        print(f"   {width_m * 100:.0f} cm block at {depth_m * 100:.0f} cm  ->  {w:.1f} px wide")
    print("   Nothing in the image separates these. Size or depth must come from outside.")
    print()

    print("3. Depth error becomes lateral error, one for one.")
    true_x, true_z = 0.12, z
    for assumed in (true_z - 0.05, true_z - 0.02, true_z + 0.03):
        rec, err = lateral_error_from_depth_error(true_x, true_z, assumed)
        print(f"   assume {assumed * 100:5.1f} cm deep -> x reads {rec * 100:5.2f} cm"
              f"  (truth {true_x * 100:.1f} cm, error {err * 100:+.2f} cm)")
    print()

    print("4. What a nudged camera does to every pixel in the frame.")
    for shift_mm in (2, 5, 10, 30):
        du = pixel_shift_from_camera_shift(shift_mm / 1000.0, z, f)
        print(f"   camera moves {shift_mm:2d} mm  ->  scene moves {du:5.1f} px"
              f"  ({100 * du / args.width:.1f}% of frame width)")
    print()
    print("The policy has no way to tell case 4 from 'the block moved'.")


if __name__ == "__main__":
    main()
