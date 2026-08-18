"""Lesson 4.6 - log what the motors are doing to themselves, then plot it.

Holding a pose is a stall: real torque, zero speed, so every watt going in comes
back out as heat. The motors will tell you, if you ask.

  python torque_watch.py --port /dev/ttyACM0 --id kp_follower --seconds 600 \
      --out warmup.csv
      Poll temperature, load and position once a second and append to a CSV.

  python torque_watch.py --plot warmup.csv
      Draw the CSV. Needs no hardware, so you can replot on any machine.

  python torque_watch.py --list-registers --port ... --id ...
      Print which register names this LeRobot version will let you read, which
      is the fastest way to fix a NameError from a renamed register.

HONESTY NOTE. Written without an SO-101 on the bench. The register names come
from the Feetech STS control table (Present_Temperature at 63, Present_Load at
60, Present_Position at 56) and the read is attempted through several plausible
bus APIs, because the exact spelling of that call has moved between LeRobot
releases. If none of them work the script prints what the bus object DOES
offer instead of dying, which is the useful failure. The plotting path is
ordinary matplotlib and is fully exercised.

Requires: matplotlib for --plot (already a course dependency); lerobot and an
arm for logging.
"""
import argparse
import csv
import sys
import time

REGISTERS = ["Present_Temperature", "Present_Load", "Present_Position"]

# Bus read methods LeRobot has exposed, in the order worth trying.
READ_METHODS = ["sync_read", "read"]

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a"]  # the course dataviz series order
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e8e7e3"


def resolve_follower():
    """Same discovery dance as first_motion.py; see the note in that file."""
    import importlib

    # `SOFollowerRobotConfig`, not `SOFollowerConfig`: the latter is exported too but
    # lacks the RobotConfig mixin, so it has no `id` field and constructing it with
    # one raises TypeError. See the longer note in first_motion.py.
    candidates = [
        ("lerobot.robots.so_follower", "SOFollower", "SOFollowerRobotConfig"),
        ("lerobot.robots.so_follower", "SO101Follower", "SO101FollowerConfig"),
        ("lerobot.robots.so101_follower", "SO101Follower", "SO101FollowerConfig"),
        ("lerobot.common.robots.so101_follower", "SO101Follower", "SO101FollowerConfig"),
    ]
    for module_name, robot_name, config_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        robot = getattr(module, robot_name, None)
        config = getattr(module, config_name, None)
        if robot and config:
            return robot, config, f"{module_name}.{robot_name}"
    print("Could not import the SO-101 follower classes. Run first_motion.py "
          "--dry-run first, and see the CANDIDATES list at the top of that file.")
    sys.exit(1)


def read_register(bus, name):
    """Read one register across every motor, whatever this version calls it."""
    last_error = None
    for method_name in READ_METHODS:
        method = getattr(bus, method_name, None)
        if method is None:
            continue
        try:
            return method(name)
        except Exception as exc:  # noqa: BLE001 - probing an API that has moved
            last_error = exc
    raise RuntimeError(
        f"no working bus read for {name!r}; last error was {last_error!r}. "
        f"The bus object offers: {sorted(n for n in dir(bus) if not n.startswith('_'))}"
    )


def log(args):
    robot_class, config_class, found = resolve_follower()
    print(f"using {found}")
    robot = robot_class(config_class(port=args.port, id=args.id))
    robot.connect()

    period = 1.0 / args.hz
    rows = []
    started = time.time()
    try:
        while time.time() - started < args.seconds:
            stamp = round(time.time() - started, 2)
            sample = {"t": stamp}
            for register in REGISTERS:
                try:
                    values = read_register(robot.bus, register)
                except RuntimeError as exc:
                    print(exc)
                    return 1
                for motor, value in dict(values).items():
                    sample[f"{motor}.{register}"] = value
            rows.append(sample)
            hottest = max(
                (v for k, v in sample.items() if k.endswith("Present_Temperature")),
                default=None,
            )
            print(f"t={stamp:7.1f}s   hottest motor: {hottest}", end="\r", flush=True)
            time.sleep(period)
    except KeyboardInterrupt:
        print("\nstopped early")
    finally:
        robot.disconnect()
        print("\ndisconnected - torque OFF, the arm is no longer holding itself up.")

    if not rows:
        print("no samples collected")
        return 1
    fieldnames = list(rows[0])
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} samples to {args.out}")
    return 0


def plot(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        print(f"{path} is empty")
        return 1

    times = [float(row["t"]) for row in rows]
    columns = [c for c in rows[0] if c.endswith("Present_Temperature")]
    if not columns:
        print(f"{path} has no temperature columns; found {list(rows[0])}")
        return 1

    figure, axes = plt.subplots(figsize=(7.2, 4.0))
    for index, column in enumerate(sorted(columns)):
        series = [float(row[column]) for row in rows]
        axes.plot(times, series, linewidth=2, color=PALETTE[index % len(PALETTE)],
                  label=column.split(".")[0])
    axes.set_xlabel("seconds", color=INK)
    axes.set_ylabel("temperature reported by the motor", color=INK)
    axes.grid(True, color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(MUTED)
    axes.tick_params(colors=MUTED)
    axes.legend(frameon=False, labelcolor=INK)

    out = path.rsplit(".", 1)[0] + ".png"
    figure.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    print("Look for the plateau. If there is one, that is your safe "
          "continuous-duty envelope. If it is still climbing, stop the session.")
    return 0


def list_registers(args):
    robot_class, config_class, found = resolve_follower()
    print(f"using {found}")
    robot = robot_class(config_class(port=args.port, id=args.id))
    robot.connect()
    try:
        table = getattr(robot.bus, "model_ctrl_table", None) or getattr(
            robot.bus, "_model_ctrl_table", None
        )
        if table is None:
            print("no control table attribute found; the bus offers:",
                  sorted(n for n in dir(robot.bus) if not n.startswith("_")))
            return 1
        for model, registers in dict(table).items():
            print(f"\n{model}:")
            for name in sorted(registers):
                print(f"  {name}")
    finally:
        robot.disconnect()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plot", help="draw an existing CSV and exit (no hardware needed)")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--id", default="my_follower")
    parser.add_argument("--seconds", type=float, default=600.0)
    parser.add_argument("--hz", type=float, default=1.0, help="samples per second")
    parser.add_argument("--out", default="torque_watch.csv")
    parser.add_argument("--list-registers", action="store_true",
                        help="print the readable register names and exit")
    args = parser.parse_args()

    if args.plot:
        return plot(args.plot)
    if args.list_registers:
        return list_registers(args)
    return log(args)


if __name__ == "__main__":
    sys.exit(main())
