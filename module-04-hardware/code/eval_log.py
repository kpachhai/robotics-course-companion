"""Log real-robot trials and report a success rate with an honest interval.

HONESTY NOTE
------------
Nothing in this file has been run against a physical arm. It records the verdicts
you type and does statistics on them. The interval it prints is a Wilson score
interval, which is the right one for small samples of successes and failures; it
answers "how much does twenty trials actually pin down", and the answer is
usually "less than you hoped".

Usage
-----
    python eval_log.py trial --result success --checkpoint 40k
    python eval_log.py trial --result failure --stage grasp --checkpoint 40k
    python eval_log.py report
    python eval_log.py report --checkpoint 40k

State lives in `eval_log.jsonl`, one JSON object per line. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

DEFAULT_LOG = Path("eval_log.jsonl")
Z_95 = 1.96

# The stages a pick-and-place trial can die at. Rename them for your own task,
# but keep the list short: a tag you have to think about is a tag you will skip.
STAGES = ["approach", "grasp", "lift", "transport", "release", "other"]


def wilson_interval(successes: int, trials: int, z: float = Z_95) -> tuple[float, float]:
    """95% Wilson score interval for a proportion.

    The normal approximation collapses at 0 successes or 100% success, which is
    exactly where a first policy lives, so use Wilson instead.
    """
    if trials == 0:
        return (0.0, 1.0)
    proportion = successes / trials
    denominator = 1 + z * z / trials
    centre = (proportion + z * z / (2 * trials)) / denominator
    spread = z / denominator * math.sqrt(
        proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)
    )
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def append(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def load(path: Path, checkpoint: str | None = None) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if checkpoint is None:
        return records
    return [record for record in records if record["checkpoint"] == checkpoint]


def report(path: Path, checkpoint: str | None) -> None:
    records = load(path, checkpoint)
    if not records:
        print(f"no trials logged in {path}" + (f" for checkpoint {checkpoint}" if checkpoint else ""))
        return

    trials = len(records)
    successes = sum(1 for record in records if record["success"])
    low, high = wilson_interval(successes, trials)

    label = checkpoint or "all checkpoints"
    print(f"{label}: {successes}/{trials} = {successes / trials:.0%}")
    print(f"95% interval: {low:.0%} to {high:.0%}  (width {high - low:.0%})")

    if trials < 20:
        print(f"  ! {trials} trials is below the twenty-trial floor. The interval above is "
              "too wide to compare anything against.")

    failures = [record for record in records if not record["success"]]
    if failures:
        print("\nwhere the trials died")
        for stage, count in Counter(record["stage"] for record in failures).most_common():
            share = count / len(failures)
            print(f"  {count:>3}  {share:>4.0%}  {stage}")
        top_stage, top_count = Counter(record["stage"] for record in failures).most_common(1)[0]
        if top_count / len(failures) >= 0.5:
            print(f"\n  more than half of your failures are at '{top_stage}'.")
            print("  that is a single stage, so target the next batch of episodes at it")
            print("  rather than recording more of the whole task.")

    if checkpoint is None:
        by_checkpoint = sorted({record["checkpoint"] for record in records})
        if len(by_checkpoint) > 1:
            print("\nper checkpoint")
            for name in by_checkpoint:
                subset = load(path, name)
                hits = sum(1 for record in subset if record["success"])
                lo, hi = wilson_interval(hits, len(subset))
                print(f"  {name:>8}  {hits:>3}/{len(subset):<3} = {hits / len(subset):>4.0%}"
                      f"   [{lo:.0%}, {hi:.0%}]")
            print("\n  overlapping intervals mean you cannot yet tell these apart.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    sub = parser.add_subparsers(dest="command", required=True)

    trial = sub.add_parser("trial", help="log one trial")
    trial.add_argument("--result", required=True, choices=["success", "failure"])
    trial.add_argument("--stage", default="other", choices=STAGES,
                       help="for a failure, the stage it died at")
    trial.add_argument("--checkpoint", required=True, help="which checkpoint you are evaluating")
    trial.add_argument("--cell", default="", help="which start cell the object was in")
    trial.add_argument("--note", default="")

    show = sub.add_parser("report", help="print success rate and failure breakdown")
    show.add_argument("--checkpoint", default=None)

    args = parser.parse_args()

    if args.command == "report":
        report(args.log, args.checkpoint)
        return

    success = args.result == "success"
    record = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checkpoint": args.checkpoint,
        "success": success,
        "stage": "none" if success else args.stage,
        "cell": args.cell,
        "note": args.note,
    }
    append(args.log, record)
    records = load(args.log, args.checkpoint)
    hits = sum(1 for item in records if item["success"])
    print(f"logged {args.result} -> {args.checkpoint}: {hits}/{len(records)}")


if __name__ == "__main__":
    main()
