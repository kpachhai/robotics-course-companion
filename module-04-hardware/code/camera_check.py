"""Lesson 4.9 - check that each camera actually gives you what you asked for.

WARNING: this script has never been run against a physical camera by the
author of this course. It is written from the OpenCV and LeRobot APIs and is
deliberately small enough to read in one sitting. Treat it as a starting
point, read it before you run it, and trust your own output over this text.

What it does, per camera:
  * opens the device and requests a resolution and frame rate
  * reports what the driver actually negotiated, which is often not what you
    asked for and is where the LeRobot error
    "frame width=... or height=... do not match configured width=..." comes from
  * measures the achieved frame rate over a few seconds of real capture
  * reports mean brightness, and how much it drifts while nothing moves
    (that drift is auto-exposure, and it is a data-quality problem)
  * saves one reference frame per camera for reference_frame_match.py

Needs opencv-python, which LeRobot already installs with the hardware extra:
    pip install opencv-python

Run:  python camera_check.py --cameras 0 1
      python camera_check.py --cameras /dev/video0 /dev/video2 --seconds 10
"""
import argparse
import time
from pathlib import Path

try:
    import cv2
except ImportError as exc:  # fail loud, not later and confusingly
    raise SystemExit("opencv-python is not installed: pip install opencv-python") from exc

import numpy as np

REFERENCE_DIR = Path(__file__).parent / "reference_frames"


def open_camera(spec, width, height, fps):
    """Open a camera by integer index or by device path."""
    source = int(spec) if str(spec).isdigit() else str(spec)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"could not open camera {spec!r}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    return capture


def measure(capture, seconds):
    """Capture for a while. Return (frames, achieved_fps, brightness series)."""
    frames_seen = 0
    brightness = []
    last_frame = None
    started = time.perf_counter()
    while time.perf_counter() - started < seconds:
        ok, frame = capture.read()
        if not ok:
            raise SystemExit("camera stopped returning frames mid-measurement")
        frames_seen += 1
        last_frame = frame
        brightness.append(float(frame.mean()))
    elapsed = time.perf_counter() - started
    return frames_seen, frames_seen / elapsed, np.array(brightness), last_frame


def report(spec, capture, requested, seconds):
    width_req, height_req, fps_req = requested
    width_got = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_got = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_claimed = capture.get(cv2.CAP_PROP_FPS)

    frames, fps_measured, brightness, frame = measure(capture, seconds)

    print(f"camera {spec}")
    print(f"  requested   {width_req}x{height_req} @ {fps_req}")
    print(f"  negotiated  {width_got}x{height_got} @ {fps_claimed:.1f} (driver's claim)")
    print(f"  measured    {fps_measured:.1f} fps over {frames} frames")
    print(f"  brightness  mean {brightness.mean():.1f}, "
          f"range {brightness.min():.1f} to {brightness.max():.1f}")

    if (width_got, height_got) != (width_req, height_req):
        print("  MISMATCH: the driver ignored your resolution. LeRobot will refuse this.")
    if fps_measured < 0.85 * fps_req:
        print("  SLOW: well under the requested rate. Suspect USB bandwidth or exposure.")
    if brightness.max() - brightness.min() > 4.0:
        print("  DRIFTING: brightness moved while nothing did. Auto-exposure is hunting.")

    REFERENCE_DIR.mkdir(exist_ok=True)
    name = str(spec).replace("/", "_").strip("_")
    path = REFERENCE_DIR / f"{name}.png"
    cv2.imwrite(str(path), frame)
    print(f"  reference frame saved to {path}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cameras", nargs="+", required=True,
                        help="indices (0 1) or device paths (/dev/video0)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=5.0)
    args = parser.parse_args()

    print("Hold still and change nothing while this runs.\n")
    for spec in args.cameras:
        capture = open_camera(spec, args.width, args.height, args.fps)
        try:
            report(spec, capture, (args.width, args.height, args.fps), args.seconds)
        finally:
            capture.release()

    print("Now open both cameras at once and run this again with both listed.")
    print("A camera that passes alone and fails in company is a bus problem.")


if __name__ == "__main__":
    main()
