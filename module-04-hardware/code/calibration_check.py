"""Lesson 4.4 - read your own calibration back, and turn it into an error budget.

Two modes, neither of which needs the arm plugged in:

  python calibration_check.py --file ~/.cache/huggingface/lerobot/calibration/\
robots/so_follower/kp_follower.json
      Print the map: homing offset, range ends, swept span per joint, and a
      warning on any joint whose sweep looks short next to its siblings.

  python calibration_check.py --budget 0.30,0.26,0.17,0.08,0.05,0.03
      Distances in metres from each joint axis to the gripper, measured on YOUR
      arm with a tape. Prints millimetres of tip error per degree of
      calibration error at that joint.

  python calibration_check.py --budget 0.30,0.26 --miss 30
      Same, plus: how many degrees of error at each joint would explain a miss
      of 30 mm.

HONESTY NOTE. This course was written without an SO-101 on the bench, so the
JSON parsing here is defensive on purpose: the file layout was inferred from
LeRobot's MotorCalibration fields (id, drive_mode, homing_offset, range_min,
range_max) rather than read off a real file. If your file has a shape this does
not recognise, the script says so and prints what it did find, which is more
useful than a traceback. The arithmetic in --budget mode is exact and owes
nothing to hardware.

Stdlib only. No dependencies beyond Python itself.
"""
import argparse
import json
import statistics
import sys

COUNTS_PER_TURN = 4096
DEG_PER_COUNT = 360.0 / COUNTS_PER_TURN

# The fixed name-to-id map baked into the SO-101 follower. Printing in this
# order rather than JSON order keeps the table readable base-outward.
JOINT_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# wrist_roll is excluded from the calibration sweep and hardcoded to the full
# turn because it rotates continuously, so a "short sweep" warning about it
# would be noise.
CONTINUOUS = {"wrist_roll"}

FIELDS = ("id", "homing_offset", "range_min", "range_max")


def find_motor_table(blob):
    """Return {motor_name: {field: value}} from whatever the file wraps it in."""
    if isinstance(blob, dict) and any(
        isinstance(v, dict) and "homing_offset" in v for v in blob.values()
    ):
        return {k: v for k, v in blob.items() if isinstance(v, dict)}
    # A future release could nest it; look one level down before giving up.
    if isinstance(blob, dict):
        for value in blob.values():
            if isinstance(value, dict):
                found = find_motor_table(value)
                if found:
                    return found
    return {}


def show_file(path):
    with open(path) as handle:
        blob = json.load(handle)

    motors = find_motor_table(blob)
    if not motors:
        print(f"Could not find calibration entries in {path}.")
        print("Top-level keys were:", list(blob) if isinstance(blob, dict) else type(blob))
        print("Open the file and compare against the fields this script expects:", FIELDS)
        return 1

    ordered = [n for n in JOINT_ORDER if n in motors]
    ordered += [n for n in motors if n not in JOINT_ORDER]

    print(f"{path}\n")
    header = f"{'joint':<16}{'id':>4}{'homing':>10}{'min':>8}{'max':>8}{'span':>8}{'span°':>9}"
    print(header)
    print("-" * len(header))

    spans = {}
    for name in ordered:
        row = motors[name]
        lo, hi = row.get("range_min"), row.get("range_max")
        span = (hi - lo) if (isinstance(lo, int) and isinstance(hi, int)) else None
        if span is not None and name not in CONTINUOUS:
            spans[name] = span
        print(
            f"{name:<16}{row.get('id', '?'):>4}{row.get('homing_offset', '?'):>10}"
            f"{lo if lo is not None else '?':>8}{hi if hi is not None else '?':>8}"
            f"{span if span is not None else '?':>8}"
            f"{span * DEG_PER_COUNT if span is not None else float('nan'):>9.1f}"
        )

    print()
    if "wrist_roll" in motors:
        row = motors["wrist_roll"]
        full = row.get("range_min") == 0 and row.get("range_max") == COUNTS_PER_TURN - 1
        print(
            "wrist_roll spans the full turn as expected."
            if full
            else "wrist_roll is NOT set to the full turn 0-4095. That is unusual; "
            "it is meant to be hardcoded, not swept."
        )

    # Heuristic, not a rule: a joint swept much less far than its siblings is
    # usually a timid sweep rather than a real mechanical limit.
    if len(spans) >= 3:
        typical = statistics.median(spans.values())
        short = {n: s for n, s in spans.items() if s < 0.6 * typical}
        if short:
            print("\nPossible short sweeps (well under the median span of "
                  f"{typical:.0f} counts):")
            for name, span in short.items():
                print(f"  {name}: {span} counts ({span * DEG_PER_COUNT:.0f}°). "
                      "Check whether the joint really stops there.")
        else:
            print("No joint looks obviously under-swept.")
    return 0


def show_budget(distances, miss_mm=None):
    print("Tip error caused by a calibration error at each joint.\n")
    header = f"{'joint':<16}{'r (m)':>8}{'mm / degree':>14}{'mm / count':>13}"
    if miss_mm is not None:
        header += f"{'° for ' + str(miss_mm) + ' mm':>16}"
    print(header)
    print("-" * len(header))

    for index, radius in enumerate(distances):
        name = JOINT_ORDER[index] if index < len(JOINT_ORDER) else f"joint_{index + 1}"
        # Small-angle: arc length = r * theta, exact enough well past 10 degrees.
        mm_per_degree = radius * 3.141592653589793 / 180.0 * 1000.0
        line = f"{name:<16}{radius:>8.3f}{mm_per_degree:>14.2f}" \
               f"{mm_per_degree * DEG_PER_COUNT:>13.3f}"
        if miss_mm is not None:
            line += f"{miss_mm / mm_per_degree:>16.2f}"
        print(line)

    print("\nOne encoder count is "
          f"{DEG_PER_COUNT:.4f}°, so the encoder is never your dominant error.")
    print("Eyeballing 'the middle of its range' to within 3° is the realistic one.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", help="path to a LeRobot calibration JSON file")
    parser.add_argument("--budget", help="comma-separated joint-to-gripper distances in metres")
    parser.add_argument("--miss", type=float, help="a miss in mm to explain, for --budget mode")
    args = parser.parse_args()

    if not args.file and not args.budget:
        parser.print_help()
        return 2

    status = 0
    if args.file:
        status |= show_file(args.file)
    if args.budget:
        if args.file:
            print()
        distances = [float(part) for part in args.budget.split(",") if part.strip()]
        status |= show_budget(distances, args.miss)
    return status


if __name__ == "__main__":
    sys.exit(main())
