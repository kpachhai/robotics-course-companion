"""Lessons 4.9 and 4.10 - has the rig moved since you recorded?

WARNING: never run against a physical camera by the author of this course.
Read it before you trust it.

The idea. Your policy learned a mapping from pixels to joint targets, so the
camera pose and the lighting at evaluation time have to match the ones you
recorded under. There is no calibration file to check, because there is no
calibration. What you have instead is a picture of the world as it looked on
recording day. This script scores a live frame against that picture.

Two scores, deliberately separate, because they fail for different reasons:

  geometry   mean absolute difference after both frames are normalised to the
             same mean brightness. Sensitive to the camera having moved.
  exposure   difference in mean brightness. Sensitive to the light having
             changed while the camera stayed put.

Both are crude. They are not a measurement of anything physical; they are a
tripwire. A score that jumps after you bump the desk is doing its job.

Run:  python reference_frame_match.py --camera 0 \
          --reference reference_frames/0.png --live
"""
import argparse
import time
from pathlib import Path

try:
    import cv2
except ImportError as exc:
    raise SystemExit("opencv-python is not installed: pip install opencv-python") from exc

import numpy as np


def to_gray_float(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)


def scores(reference, live):
    """Return (geometry_score, exposure_delta).

    geometry_score is mean absolute difference in grey levels after matching
    mean brightness, so a pure lighting change does not inflate it.
    """
    ref = to_gray_float(reference)
    now = to_gray_float(live)
    if ref.shape != now.shape:
        raise SystemExit(f"shape mismatch: reference {ref.shape}, live {now.shape}")
    exposure_delta = float(now.mean() - ref.mean())
    geometry = float(np.abs((now - now.mean()) - (ref - ref.mean())).mean())
    return geometry, exposure_delta


def verdict(geometry, exposure_delta):
    """Thresholds are guesses. Calibrate them on your own desk: record a
    baseline with nothing touched, then deliberately nudge the camera 5 mm and
    see what the number does. Replace these two constants with what you saw."""
    notes = []
    if geometry > 12.0:
        notes.append("geometry: something moved")
    if abs(exposure_delta) > 8.0:
        notes.append("exposure: the light changed")
    return ", ".join(notes) if notes else "match"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", required=True, help="index or device path")
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--live", action="store_true",
                        help="keep printing, so you can nudge the camera and watch")
    args = parser.parse_args()

    reference = cv2.imread(str(args.reference))
    if reference is None:
        raise SystemExit(f"could not read reference image {args.reference}")

    source = int(args.camera) if str(args.camera).isdigit() else str(args.camera)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"could not open camera {args.camera!r}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise SystemExit("camera stopped returning frames")
            geometry, exposure_delta = scores(reference, frame)
            print(f"geometry {geometry:6.2f}   exposure {exposure_delta:+6.2f}   "
                  f"{verdict(geometry, exposure_delta)}")
            if not args.live:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()


if __name__ == "__main__":
    main()
