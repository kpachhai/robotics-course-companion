"""Lessons 4.2-4.3 - generate a build sheet you fill in with a pen.

The build has an ordering constraint (a motor is configured before it is
installed) and four gates that fail silently if skipped. A written sheet is the
cheapest defence: it makes the order visible and it becomes the record you read
back when something misbehaves three lessons later.

Runs anywhere. No hardware, no dependencies beyond the standard library. The ID
plan it prints is LeRobot's hardcoded name-to-ID mapping for the SO-101, read
from the library source on 2026-08-09; the configuration order is reversed
because that is the order the setup tool walks.

Run:  python build_sheet.py
      python build_sheet.py --robot-id my_follower --teleop-id my_leader
      python build_sheet.py --output build-sheet.md
"""
import argparse
import datetime as dt

# LeRobot's fixed mapping. shoulder_pan is ID 1 through gripper at ID 6, and
# the gripper is the one joint reported as 0-100 rather than in degrees.
JOINTS = [
    ("shoulder_pan", 1, "degrees"),
    ("shoulder_lift", 2, "degrees"),
    ("elbow_flex", 3, "degrees"),
    ("wrist_flex", 4, "degrees"),
    ("wrist_roll", 5, "degrees, full-turn joint"),
    ("gripper", 6, "0 to 100"),
]

GATES = [
    ("Print accuracy",
     "Gauge_0 / Gauge_tight_1 fit a real STS3215, or the Lego gauges fit a 4x2 brick",
     "skipping it costs a reprint of both arms"),
    ("Motor identity",
     "every motor has its ID written before it goes into a printed part",
     "skipping it costs partial disassembly"),
    ("Orientation",
     "each horn and link seated as the assembly video shows",
     "skipping it costs a joint whose usable range sits in the wrong place"),
    ("Cable slack",
     "a loop of spare cable at every joint that rotates",
     "skipping it costs a motor that intermittently drops off the bus"),
]

LED_TRIAGE = [
    ("steady red, gripper to base", "wiring is fine; look at software"),
    ("one or more dark, chain stops", "reseat the 3-pin cables, check the board supply"),
    ("blinking", "error state: overload, or the wrong supply voltage"),
]


def sheet(robot_id: str, teleop_id: str) -> str:
    today = dt.date.today().isoformat()
    out = [
        f"# SO-101 build sheet ({today})",
        "",
        "Fill this in with a pen as you go. It is the record you read back when",
        "something misbehaves later in the module.",
        "",
        "## Identifiers, chosen once and never changed",
        "",
        "| What | Value |",
        "|---|---|",
        f"| follower `--robot.id` | `{robot_id}` |",
        f"| leader `--teleop.id` | `{teleop_id}` |",
        "| follower port | `______________________` |",
        "| leader port | `______________________` |",
        "",
        "These strings are how every later command finds your calibration.",
        "Change one and the tools quietly find nothing and start calibration again.",
        "",
        "## Gate log",
        "",
        "| Gate | Passed when | Cost of skipping | Done |",
        "|---|---|---|---|",
    ]
    for name, passes, cost in GATES:
        out.append(f"| {name} | {passes} | {cost} | [ ] |")

    out += [
        "",
        "### Gate 1 notes: printer settings that got the gauge to fit",
        "",
        "```",
        "nozzle / layer     : ____________________",
        "flow or expansion  : ____________________",
        "attempts needed    : ____________________",
        "```",
        "",
        "## Motor configuration, in the order the tool walks",
        "",
        "One motor connected at a time. Check the cabling before every Enter.",
        "",
        "| Order | Joint | ID | Reported as | Follower | Leader |",
        "|---|---|---|---|---|---|",
    ]
    for position, (name, motor_id, units) in enumerate(reversed(JOINTS), start=1):
        out.append(f"| {position} | `{name}` | {motor_id} | {units} | [ ] | [ ] |")

    out += [
        "",
        "## Assembly, joint by joint",
        "",
        "| Joint | Motor configured | Horn orientation checked | Cable slack left | Swept by hand |",
        "|---|---|---|---|---|",
    ]
    for name, _motor_id, _units in JOINTS:
        out.append(f"| `{name}` | [ ] | [ ] | [ ] | [ ] |")

    out += [
        "",
        "## Acceptance test",
        "",
        "Power the chain and read the LEDs before opening any log file.",
        "",
        "| What you see | What it means |",
        "|---|---|",
    ]
    for symptom, meaning in LED_TRIAGE:
        out.append(f"| {symptom} | {meaning} |")

    out += [
        "",
        "Most 'timeout' errors are physical, not code.",
        "",
        "## Anything surprising",
        "",
        "Write it down here while it is still fresh. Future you is the reader.",
        "",
        "- ",
        "- ",
        "- ",
        "",
    ]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--robot-id", default="my_follower",
                        help="the --robot.id string you will use for the follower, forever")
    parser.add_argument("--teleop-id", default="my_leader",
                        help="the --teleop.id string you will use for the leader, forever")
    parser.add_argument("--output", help="write the sheet to this file instead of stdout")
    args = parser.parse_args()

    text = sheet(args.robot_id, args.teleop_id)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)
        print(f"wrote {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
