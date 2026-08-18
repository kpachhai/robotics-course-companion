"""Keep score of a recording session: what you kept, what you threw away, and why.

HONESTY NOTE
------------
This is a notebook, not an instrument. It records what you tell it. It never
reads the dataset and never talks to the robot, so it cannot catch a lie. Its
whole value is that it makes the keep rate visible while you can still do
something about it.

Usage
-----
    python session_log.py keep --cell A1
    python session_log.py discard --cell A2 --reason grasp-missed
    python session_log.py report

State lives in `session_log.jsonl` next to wherever you run it, one JSON object
per line. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

DEFAULT_LOG = Path("session_log.jsonl")
WINDOW = 10


def append(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def keep_rate(records: list[dict]) -> float:
    if not records:
        return 0.0
    kept = sum(1 for record in records if record["kept"])
    return kept / len(records)


def report(path: Path) -> None:
    records = load(path)
    if not records:
        print(f"no episodes logged yet in {path}")
        return

    kept = [record for record in records if record["kept"]]
    discarded = [record for record in records if not record["kept"]]

    print(f"attempts   {len(records)}")
    print(f"kept       {len(kept)}")
    print(f"discarded  {len(discarded)}")
    print(f"keep rate  {keep_rate(records):.0%}  (last {WINDOW}: {keep_rate(records[-WINDOW:]):.0%})")

    if discarded:
        print("\nwhy episodes were thrown away")
        for reason, count in Counter(r["reason"] for r in discarded).most_common():
            print(f"  {count:>3}  {reason}")

    print("\nkept episodes per start cell")
    per_cell = Counter(record["cell"] for record in kept)
    if per_cell:
        for cell, count in sorted(per_cell.items()):
            print(f"  {cell:>4}  {count:>3}  {'#' * count}")
        gap = max(per_cell.values()) - min(per_cell.values())
        if gap > 2:
            print(f"  ! coverage is uneven: {gap} episodes between the fullest and "
                  "emptiest cell")
    else:
        print("  none yet")

    if len(records) >= WINDOW and keep_rate(records[-WINDOW:]) < keep_rate(records) - 0.15:
        print("\n! your keep rate over the last ten is well below your session average.")
        print("  that usually means tired hands rather than a harder cell. Take a break;")
        print("  sloppy demonstrations are worse than fewer demonstrations.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    sub = parser.add_subparsers(dest="command", required=True)

    keep = sub.add_parser("keep", help="log an episode you kept")
    keep.add_argument("--cell", required=True)
    keep.add_argument("--note", default="")

    discard = sub.add_parser("discard", help="log an episode you threw away")
    discard.add_argument("--cell", required=True)
    discard.add_argument("--reason", required=True,
                         help="one of your own tags, e.g. grasp-missed, hesitated, dropped")
    discard.add_argument("--note", default="")

    sub.add_parser("report", help="print the running tally")

    args = parser.parse_args()

    if args.command == "report":
        report(args.log)
        return

    record = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cell": args.cell,
        "kept": args.command == "keep",
        "reason": getattr(args, "reason", "kept"),
        "note": args.note,
    }
    append(args.log, record)
    records = load(args.log)
    verb = "kept" if record["kept"] else f"discarded ({record['reason']})"
    print(f"{verb} in {record['cell']}  ->  {len(records)} attempts, "
          f"keep rate {keep_rate(records):.0%}")


if __name__ == "__main__":
    main()
