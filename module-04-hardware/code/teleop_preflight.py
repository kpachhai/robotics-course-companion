"""Lesson 4.7 - keep one set of ports and ids, and never type them by hand again.

The identifier you pass as --robot.id / --teleop.id is the primary key into your
calibration. Type it differently once and LeRobot finds no calibration file, says
so quietly, and starts asking you to sweep joints again. This script stores the
values once and prints the commands with them substituted, so the strings cannot
drift between calibrate, teleoperate, record and rollout.

    python teleop_preflight.py init        # answer six questions, writes JSON
    python teleop_preflight.py check       # the physical checklist, before software
    python teleop_preflight.py commands    # print your commands, ready to paste

Config file: ./so101_setup.json, or pass --config PATH.

NO HARDWARE IS TOUCHED HERE. This script opens no serial port and imports no
LeRobot code; it stores strings and formats commands. Everything it prints was
transcribed from the LeRobot docs for the v0.6.x line - check them against the
docs for the version you actually installed before pasting.
"""

import argparse
import json
import sys
from pathlib import Path

DEFAULT_CONFIG = Path("so101_setup.json")

FIELDS = [
    ("follower_port", "Follower serial port", "/dev/ttyACM0"),
    ("leader_port", "Leader serial port", "/dev/ttyACM1"),
    ("follower_id", "Follower id (the calibration key, pick once, keep forever)", "my_follower"),
    ("leader_id", "Leader id", "my_leader"),
    ("top_camera", "Fixed camera index or path (blank for none)", "1"),
    ("wrist_camera", "Wrist camera index or path (blank for none)", "0"),
]

# Ordered physical checks. Each is (prompt, what to do when the answer is no).
CHECKLIST = [
    (
        "Follower and leader are both on their own 5V/7.4V supply, and no 12V "
        "supply is anywhere near either of them",
        "Stop. A 12V supply on a 7.4V arm puts motors into an error state and can "
        "damage them. The SO-101 leader is always 7.4V.",
    ),
    (
        "If you are on a Waveshare control board: both jumpers are on the B (USB) channel",
        "Move them. Wrong channel is a silent failure - the port opens and no motor answers.",
    ),
    (
        "Every 3-pin cable is fully clicked in, board included",
        "Reseat each one. Most 'timeout' errors are physical, not code.",
    ),
    (
        "Motor LEDs: a steady red chain runs the whole way from gripper to base",
        "If the chain goes dark part way, the fault is at the last lit motor: reseat "
        "that cable, check board power. If any LED is blinking, that motor is in an "
        "error state, usually overload or the wrong supply voltage. Power-cycle after "
        "moving the joint back inside its range.",
    ),
    (
        "The follower's power plug is loose in its socket and within reach of your free hand",
        "Make it so. There is no e-stop on this arm; that plug is the e-stop. Remember "
        "that pulling it drops the arm rather than freezing it.",
    ),
    (
        "The swept volume is clear: no mug, no keyboard, nothing fragile, nothing of yours",
        "Clear it. The arm has no collision detection and no idea where your hand is.",
    ),
]


def load(path):
    if not path.exists():
        sys.exit(f"no config at {path}. Run: python {Path(__file__).name} init")
    return json.loads(path.read_text())


def cmd_init(args):
    path = Path(args.config)
    existing = json.loads(path.read_text()) if path.exists() else {}
    cfg = {}
    print("Press Enter to accept the value in brackets.\n")
    for key, prompt, default in FIELDS:
        current = existing.get(key, default)
        answer = input(f"{prompt} [{current}]: ").strip()
        cfg[key] = answer if answer else current
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"\nwrote {path}")
    print("Use the same ids for the rest of this module. Back this file up with your "
          "calibration JSON.")


def cmd_check(args):
    print("Physical pre-flight. Answer honestly; the arm does not grade on effort.\n")
    for i, (question, remedy) in enumerate(CHECKLIST, 1):
        answer = input(f"{i}. {question}?  [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print(f"\n   -> {remedy}\n")
            print("Fix that, then run this again from the top.")
            return 1
    print("\nAll clear. Now you may open a serial port.")
    return 0


def _cameras_arg(cfg):
    entries = []
    for name, key in (("top", "top_camera"), ("wrist", "wrist_camera")):
        value = str(cfg.get(key, "")).strip()
        if value:
            entries.append(
                f"{name}: {{type: opencv, index_or_path: {value}, "
                f"width: 640, height: 480, fps: 30}}"
            )
    if not entries:
        return None
    return '--robot.cameras="{ ' + ", ".join(entries) + ' }"'


def cmd_commands(args):
    cfg = load(Path(args.config))
    cams = _cameras_arg(cfg)

    print("# 1. Calibrate. Run once per arm, and any time you replace a servo or")
    print("#    have a hard collision. Sweep every joint except wrist_roll.")
    print("lerobot-calibrate --robot.type=so101_follower \\")
    print(f"    --robot.port={cfg['follower_port']} --robot.id={cfg['follower_id']}")
    print()
    print("lerobot-calibrate --teleop.type=so101_leader \\")
    print(f"    --teleop.port={cfg['leader_port']} --teleop.id={cfg['leader_id']}")
    print()
    print("# 2. Teleoperate. Start with no cameras at all, then add them.")
    print("lerobot-teleoperate \\")
    print(f"    --robot.type=so101_follower --robot.port={cfg['follower_port']} "
          f"--robot.id={cfg['follower_id']} \\")
    print(f"    --teleop.type=so101_leader --teleop.port={cfg['leader_port']} "
          f"--teleop.id={cfg['leader_id']}")
    print()
    if cams:
        print("# 3. Same thing with cameras and the live view.")
        print("lerobot-teleoperate \\")
        print(f"    --robot.type=so101_follower --robot.port={cfg['follower_port']} "
              f"--robot.id={cfg['follower_id']} \\")
        print(f"    {cams} \\")
        print(f"    --teleop.type=so101_leader --teleop.port={cfg['leader_port']} "
              f"--teleop.id={cfg['leader_id']} \\")
        print("    --display_data=true")
        print()
    print("# Recording and deployment take the same two id flags, unchanged.")
    print("# If a command starts asking you to sweep joints, the id did not match a")
    print("# calibration file. Quit, fix the string, do not sweep.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=["init", "check", "commands"])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = ap.parse_args()
    return {"init": cmd_init, "check": cmd_check, "commands": cmd_commands}[args.action](args)


if __name__ == "__main__":
    sys.exit(main() or 0)
