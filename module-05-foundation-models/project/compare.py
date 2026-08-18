"""Module 5 milestone - the comparison (Lesson 5.18).

A harness for an honest head-to-head between policies on ONE robot, ONE
dataset and ONE evaluation protocol. This is a scaffold, not a starter with
blanks: the experiment is yours to design. No solution file exists for this
one, deliberately - a "solution" would be someone else's numbers on someone
else's bench, which is exactly the thing that does not transfer.

What is given, because a silent error here would poison the whole project:
  - randomised, interleaved trial ordering (protocol hygiene, and boring)
  - an append-one-row-per-trial recorder (scoring from memory is the enemy)
  - a correct Wilson confidence interval, with a self-check you can run

What is yours (marked TODO(you) / NotImplementedError):
  - the task definitions, including the success criterion you must write down
  - the adapter that invokes your policies, which depends on your rig
  - what "recovery" means for your tasks

Run:  python compare.py check
      python compare.py plan   --seed 7 --trials 20
      python compare.py record --seed 7 --trials 20 --out results.csv
      python compare.py report --in results.csv

Stdlib only. No robot required to run `check`, `plan` or `report`.
"""
import argparse
import csv
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------
# The experiment. Edit this block; it IS the design.
# --------------------------------------------------------------------------

POLICIES = ["act", "smolvla"]  # TODO(you): add "pi05" if you have >=24 GB VRAM

TASKS = {
    # id: (instruction sent to the policy, what the OOD variant changes)
    "place":     ("put the block in the cup", "block colour never seen in training"),
    "place_rand": ("put the block in the cup", "block starts outside the trained region"),
    "select":    ("pick the red block", "a fourth distractor colour is added"),
    "two_step":  ("put the block in the cup, then put the cup on the coaster", None),
}

# TODO(you): write these out, in a file you commit, BEFORE training anything.
#   reset_spec[task]        - tape marks, photo, tolerance. How the scene is rebuilt.
#   success_criterion[task] - the sentence the scorer reads. Not an impression.
# A criterion you did not write down is a vibe, and a vibe drifts over 400 trials.
RESET_SPEC = {}          # task id -> path to your written spec / photo
SUCCESS_CRITERION = {}   # task id -> the verbatim sentence

CONDITIONS = ["id", "ood"]          # in-distribution, out-of-distribution
TIMEOUT_S = 60                       # one timeout, applied to every policy
FAILURE_MODES = [
    "none",          # trial succeeded
    "state",         # went to the wrong object or the wrong place
    "grasp",         # grasped and dropped, or slipped
    "loop",          # repeated the same motion until the timeout
    "other",
]

CSV_FIELDS = [
    "logged_at", "seed", "index", "policy", "task", "condition",
    "first_attempt", "outcome", "failure_mode", "seconds", "notes",
]


# --------------------------------------------------------------------------
# Trial planning - given, because interleaving is the whole point and it is dull
# --------------------------------------------------------------------------

def build_plan(seed, trials_per_cell, policies=None, tasks=None):
    """Every (policy, task, condition) cell, repeated `trials_per_cell` times,
    shuffled into one interleaved order.

    Interleaving is not cosmetic. Run one policy's trials as a block and the
    model identity is perfectly confounded with the hour of the evening, the
    arm's temperature, and how good you have got at resetting the scene.
    """
    policies = policies or POLICIES
    tasks = tasks or list(TASKS)
    cells = [
        (policy, task, condition)
        for policy in policies
        for task in tasks
        for condition in CONDITIONS
        # the two-step task has no OOD variant defined; skip that cell
        if not (condition == "ood" and TASKS[task][1] is None)
    ]
    plan = [cell for cell in cells for _ in range(trials_per_cell)]
    random.Random(seed).shuffle(plan)
    return plan


# --------------------------------------------------------------------------
# Statistics - given, because a quietly wrong interval is worse than none
# --------------------------------------------------------------------------

def wilson_interval(successes, n, z=1.96):
    """95% Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because it stays inside [0, 1] and
    stays useful at the edges, which is where a real-robot table lives. Nought
    successes in twenty trials is NOT "0%"; it is "below about 16%".
    """
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def recovery_rate(rows):
    """Of the trials whose FIRST attempt failed, what fraction finished?

    TODO(you): define "first attempt" for your tasks and implement this over
    the recorded rows. It is three lines once you have decided; deciding is the
    work. On the independent SO-101 benchmark this number separated the models
    far more cleanly than success rate did (about 6% against about 31%).
    """
    raise NotImplementedError("define what a first attempt is for your tasks")


# --------------------------------------------------------------------------
# The policy adapter - yours, because it depends on your inference stack
# --------------------------------------------------------------------------

def run_policy_trial(policy, task, condition):
    """Drive one episode autonomously and return (outcome, seconds).

    TODO(you): wire this to however you serve your fine-tuned checkpoints.
    Honour TIMEOUT_S. Until this exists, use `record` without --auto and score
    the trials by hand, which is what most published real-robot numbers are
    anyway.
    """
    raise NotImplementedError("wire this to your own inference stack")


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def cmd_check(_args):
    """Verify the interval arithmetic against two worked cases from the lesson."""
    lo, hi = wilson_interval(0, 20)
    assert abs(lo - 0.0) < 1e-9, lo
    assert abs(hi - 0.161) < 0.002, hi
    lo, hi = wilson_interval(10, 20)
    assert abs(lo - 0.299) < 0.002, lo
    assert abs(hi - 0.701) < 0.002, hi
    print("wilson_interval(0, 20)  = 0.0%  to 16.1%   <- a 0/20 cell is not 'never'")
    print("wilson_interval(10, 20) = 29.9% to 70.1%   <- +/- 20 points at n=20")
    print("ok")


def cmd_plan(args):
    plan = build_plan(args.seed, args.trials)
    for index, (policy, task, condition) in enumerate(plan):
        print(f"{index:>4}  {policy:<10} {task:<12} {condition}")
    print(f"\n{len(plan)} trials, seed {args.seed}. "
          f"At ~90 s each with resets that is about {len(plan) * 90 / 3600:.1f} hours.",
          file=sys.stderr)


def cmd_record(args):
    plan = build_plan(args.seed, args.trials)
    path = Path(args.out)
    fresh = not path.exists()
    done = 0
    if not fresh:
        with path.open(newline="") as handle:
            done = sum(1 for _ in csv.DictReader(handle))
        print(f"resuming {path} at trial {done}")

    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if fresh:
            writer.writeheader()
        for index in range(done, len(plan)):
            policy, task, condition = plan[index]
            instruction = TASKS[task][0]
            print(f"\n--- trial {index + 1}/{len(plan)} "
                  f"| {policy} | {task} | {condition} ---")
            print(f"    reset the scene per {RESET_SPEC.get(task, 'TODO(you)')}")
            print(f"    instruction: {instruction!r}   timeout: {TIMEOUT_S} s")
            print(f"    criterion:   {SUCCESS_CRITERION.get(task, 'TODO(you)')}")

            if args.auto:
                outcome, seconds = run_policy_trial(policy, task, condition)
                first = outcome
            else:
                first = _ask("first attempt succeeded? [y/n]", {"y", "n"})
                outcome = _ask("trial finished successfully? [y/n]", {"y", "n"})
                seconds = _ask_float("seconds elapsed")

            mode = "none" if outcome == "y" else _ask(
                f"failure mode {FAILURE_MODES[1:]}", set(FAILURE_MODES))
            writer.writerow({
                "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "seed": args.seed, "index": index, "policy": policy,
                "task": task, "condition": condition,
                "first_attempt": first, "outcome": outcome,
                "failure_mode": mode, "seconds": seconds,
                "notes": input("notes (enter to skip): ").strip(),
            })
            handle.flush()   # one row per trial, on disk, before the next reset


def cmd_report(args):
    with Path(args.infile).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        sys.exit("no rows")

    print("| policy | task | cond | n | success | 95% Wilson |")
    print("|---|---|---|---|---|---|")
    for policy in sorted({r["policy"] for r in rows}):
        for task in sorted({r["task"] for r in rows}):
            for condition in CONDITIONS:
                cell = [r for r in rows if r["policy"] == policy
                        and r["task"] == task and r["condition"] == condition]
                if not cell:
                    continue
                hits = sum(1 for r in cell if r["outcome"] == "y")
                lo, hi = wilson_interval(hits, len(cell))
                print(f"| {policy} | {task} | {condition} | {len(cell)} | "
                      f"{hits / len(cell):.0%} | {lo:.0%} to {hi:.0%} |")

    # TODO(you): recovery rate per policy, dominant failure mode per policy,
    # and the cost column. The first one is a stub above and blocks this line.
    print("\nrecovery rate:", recovery_rate(rows))


def _ask(prompt, allowed):
    while True:
        value = input(f"    {prompt}: ").strip().lower()
        if value in allowed:
            return value


def _ask_float(prompt):
    while True:
        try:
            return float(input(f"    {prompt}: ").strip())
        except ValueError:
            continue


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check").set_defaults(func=cmd_check)

    for name in ("plan", "record"):
        p = sub.add_parser(name)
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--trials", type=int, default=20, help="trials per cell")
        p.set_defaults(func=cmd_plan if name == "plan" else cmd_record)
    sub.choices["record"].add_argument("--out", default="results.csv")
    sub.choices["record"].add_argument("--auto", action="store_true",
                                       help="drive the robot via run_policy_trial")

    report = sub.add_parser("report")
    report.add_argument("--in", dest="infile", default="results.csv")
    report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
