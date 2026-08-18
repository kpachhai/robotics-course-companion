"""Lesson 4.5 - the smallest command that exercises the whole chain.

Reads the arm, nudges ONE joint by a known number of degrees, then watches
where it actually settles and reports the gap.

  python first_motion.py --dry-run
      Prints the plan. Needs no hardware, no LeRobot, no arm.

  python first_motion.py --port /dev/ttyACM0 --id kp_follower --connect-only
      Connects, holds torque on for a few seconds, disconnects. This is the
      "push it and feel it stiffen" experiment from the lesson.

  python first_motion.py --port /dev/ttyACM0 --id kp_follower \
      --joint shoulder_pan --delta 5
      One joint, five degrees, then the settle report.

  ... --delta 40 --max-relative-target 5
      Same command with the per-step motion clamp switched on (lesson 4.6).

HONESTY NOTE. This course was written with no SO-101 on the bench, so none of
the hardware paths below have been run against a physical arm. Two consequences
shaped the design, and both are deliberate:

  1. Nothing here hardcodes an import path or an observation key. Class names
     and module paths have moved between LeRobot releases (0.4.x renamed the
     robot classes, 0.6.1 renamed lerobot.types), so the script discovers what
     is installed and TELLS you what it found. When a rename lands you get a
     readable message instead of an ImportError or a KeyError.
  2. Every number it prints comes from your arm. There are no expected values
     in this file, because the author has none to offer.

Requires: lerobot with the feetech extra, and an arm. Dry-run requires neither.
"""
import argparse
import dataclasses
import sys
import time

# Import paths LeRobot has used for the SO-101 follower, newest first. The
# class was renamed from SO101Follower to SOFollower during the 0.4 series and
# SO101Follower kept as an alias; both spellings are worth trying.
#
# Config-class order matters more than it looks. On 0.6.x the module also exports a
# bare `SOFollowerConfig`, which is the plain field-holder WITHOUT the RobotConfig
# mixin - so it has no `id` and no `calibration_dir`. Constructing the robot from it
# raises AttributeError on `config.id`, and even if it did not, dropping `id` would
# silently load the wrong calibration. Ask for the registered config classes only.
CANDIDATES = [
    ("lerobot.robots.so_follower", "SOFollower", "SOFollowerRobotConfig"),
    ("lerobot.robots.so_follower", "SO101Follower", "SO101FollowerConfig"),
    ("lerobot.robots.so101_follower", "SO101Follower", "SO101FollowerConfig"),
    ("lerobot.common.robots.so101_follower", "SO101Follower", "SO101FollowerConfig"),
]

# Config fields this script must not proceed without. `id` selects the calibration
# file; running with the wrong one, or with none, is the failure lesson 4 is about.
REQUIRED_CONFIG_FIELDS = {"port", "id"}


def resolve_follower():
    """Return (RobotClass, ConfigClass, description) for whatever is installed."""
    import importlib

    attempts = []
    for module_name, robot_name, config_name in CANDIDATES:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            attempts.append(f"  {module_name}: {exc}")
            continue
        robot = getattr(module, robot_name, None)
        config = getattr(module, config_name, None)
        if robot and config:
            return robot, config, f"{module_name}.{robot_name}"
        attempts.append(f"  {module_name}: imported, but no {robot_name}/{config_name}")

    print("Could not find the SO-101 follower classes in your LeRobot install.")
    print("Tried:")
    print("\n".join(dict.fromkeys(attempts)))
    print("\nFind the real path with:")
    print("  python -c \"import lerobot, pathlib; "
          "print(pathlib.Path(lerobot.__file__).parent)\"")
    print("then look under robots/ and add it to CANDIDATES at the top of this file.")
    sys.exit(1)


def build_config(config_class, **wanted):
    """Pass only the fields this version of the config declares, and refuse to
    drop the ones that change which robot or which calibration we talk to."""
    if dataclasses.is_dataclass(config_class):
        allowed = {field.name for field in dataclasses.fields(config_class)}
        dropped = sorted(set(wanted) - allowed)
        fatal = sorted(set(dropped) & REQUIRED_CONFIG_FIELDS)
        if fatal:
            print(f"{config_class.__name__} does not declare {', '.join(fatal)}.")
            print("That is the wrong config class, not a version difference: dropping "
                  "'id' would load someone else's calibration, or none at all.")
            print("Edit CANDIDATES at the top of this file to name the config class "
                  "that inherits RobotConfig in your install.")
            sys.exit(1)
        if dropped:
            print(f"note: this LeRobot version has no config field(s): {', '.join(dropped)}")
        wanted = {key: value for key, value in wanted.items() if key in allowed}
    return config_class(**wanted)


def position_keys(observation):
    """The joint-position entries of an observation, whatever the suffix is."""
    return sorted(
        key for key, value in observation.items()
        if isinstance(value, (int, float)) and key.endswith(".pos")
    )


def pick_key(keys, joint):
    exact = [key for key in keys if key.split(".")[0] == joint]
    if exact:
        return exact[0]
    print(f"No position key for joint {joint!r}. The arm reported: {keys}")
    sys.exit(1)


def send(robot, action, fallback_key):
    """Send the whole position vector; fall back to a single key if refused."""
    try:
        return robot.send_action(action)
    except Exception as exc:  # noqa: BLE001 - the shape of this API has moved
        print(f"send_action rejected the full action dict ({exc!r}); "
              f"retrying with {fallback_key} alone.")
        return robot.send_action({fallback_key: action[fallback_key]})


def run(args):
    robot_class, config_class, found = resolve_follower()
    print(f"using {found}")

    config = build_config(
        config_class,
        port=args.port,
        id=args.id,
        max_relative_target=args.max_relative_target,
    )
    robot = robot_class(config)

    robot.connect()
    try:
        if args.connect_only:
            print(f"connected, torque on. Push a joint. Disconnecting in {args.settle:.0f}s.")
            time.sleep(args.settle)
            return

        observation = robot.get_observation()
        keys = position_keys(observation)
        print(f"observation position keys: {keys}")

        key = pick_key(keys, args.joint)
        start = float(observation[key])
        target = start + args.delta
        print(f"{key}: at {start:.2f}, commanding {target:.2f} (delta {args.delta:+.2f})")

        action = {name: float(observation[name]) for name in keys}
        action[key] = target
        send(robot, action, key)

        deadline = time.time() + args.settle
        reached = start
        while time.time() < deadline:
            time.sleep(0.05)
            reached = float(robot.get_observation()[key])
        gap = target - reached

        print(f"settled at {reached:.2f}")
        print(f"gap: {gap:+.2f}° ({abs(gap) / max(abs(args.delta), 1e-9) * 100:.1f}% of the step)")
        print("That gap is your arm's number. Write it down; nothing here predicts it.")
    finally:
        robot.disconnect()
        print("disconnected - torque is now OFF, so the arm is no longer holding itself up.")


def dry_run(args):
    print("DRY RUN - nothing is opened, nothing moves.\n")
    print(f"  port                 {args.port}")
    print(f"  robot id             {args.id}   (must match your calibration exactly)")
    print(f"  joint                {args.joint}")
    print(f"  delta                {args.delta:+.2f} degrees")
    print(f"  settle window        {args.settle:.1f} s")
    print(f"  max_relative_target  {args.max_relative_target}")
    print("\nIt would then:")
    print("  1. import the SO-101 follower classes and report which path worked")
    print("  2. connect - which writes mode, P/I/D and the gripper caps, and enables torque")
    print("  3. read the observation and print the position key names it found")
    print("  4. command every joint to hold, with the chosen one offset by the delta")
    print("  5. poll until the settle window expires and report the remaining gap")
    print("  6. disconnect, which turns torque OFF and lets the arm sag")
    print("\nClear the desk around the arm before running this for real.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", default="/dev/ttyACM0", help="serial port from lerobot-find-port")
    parser.add_argument("--id", default="my_follower", help="the calibration id, exactly as calibrated")
    parser.add_argument("--joint", default="shoulder_pan", help="joint to nudge")
    parser.add_argument("--delta", type=float, default=5.0, help="degrees to add")
    parser.add_argument("--settle", type=float, default=2.0, help="seconds to watch before reporting")
    parser.add_argument("--max-relative-target", type=float, default=None,
                        help="per-step motion clamp; None (the default) means no clamp at all")
    parser.add_argument("--connect-only", action="store_true",
                        help="connect, hold, disconnect - the 'feel it stiffen' experiment")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = parser.parse_args()

    if args.dry_run:
        dry_run(args)
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
