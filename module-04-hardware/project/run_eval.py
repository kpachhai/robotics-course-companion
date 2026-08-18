"""Module 4 milestone - the honest 20-trial evaluation harness (Lesson 4.18).

A scaffold, not a starter-with-blanks. The plumbing is finished: it deals the
initial conditions in a fixed random order, prompts you trial by trial, tags
failures, appends one JSON line per trial as you go, and refuses to let you
quietly re-run the trials you did not like. What is deliberately missing is the
experimental design, because the design IS the milestone:

    TASK_NAME            what you are claiming, in one sentence
    SUCCESS_CRITERION    the sentence you will judge every trial against,
                         written BEFORE the first trial
    FAILURE_MODES        the buckets you will sort failures into
    sample_initial_condition()   the distribution you are claiming to work over

Fill those four in, and the script runs. Leave any of them as shipped and it
stops with an explanation, on purpose.

The robot is not in here. This process does not talk to LeRobot, does not open
a serial port, and does not know whether a trial succeeded. You run the rollout
in another terminal, you watch it, you type the verdict. That separation is the
point: the thing that scores the experiment must not be the thing running it.

Run:
    python run_eval.py --trials 20 --out results.jsonl
    python run_eval.py --out results.jsonl --resume
    python run_eval.py --out results.jsonl --dry-run    # deal conditions only
"""

import argparse
import json
import os
import random
import sys
import time

# ---------------------------------------------------------------------------
# TODO(you): the four decisions. Everything below this block is finished.
# ---------------------------------------------------------------------------

TASK_NAME = "TODO(you): one sentence, e.g. 'pick the red cube and drop it in the bowl'"

# Judged from outside the robot, by you, in under two seconds, with no
# 'it nearly worked' branch. Write it before trial 1 and never edit it after.
SUCCESS_CRITERION = "TODO(you): e.g. 'the cube is fully inside the bowl within 30 s'"

# Sort every failure into exactly one of these while it is fresh. Four to six
# buckets. If you find yourself needing a seventh mid-run, finish the run with
# 'other' and add the bucket for the NEXT run.
FAILURE_MODES = [
    "TODO(you): e.g. missed-grasp",
    "TODO(you): e.g. grasped-then-dropped",
    "TODO(you): e.g. never-approached",
    "TODO(you): e.g. stalled-mid-task",
    "other",
]


def sample_initial_condition(rng):
    """Return a dict describing how to set the table up for one trial.

    This function IS your claim. Whatever it can produce is the distribution
    the headline number applies to, and nothing outside it is covered.

    Keys are yours; keep them short, keep them measurable, and keep them the
    same as the ones you used when recording, or the number means nothing.
    Something like:

        return {"cell": rng.choice(["A1", "A2", "B1", "B2"]),
                "object_yaw_deg": round(rng.uniform(-45, 45)),
                "distractor": rng.random() < 0.3}
    """
    raise NotImplementedError("write your initial-condition distribution")


# ---------------------------------------------------------------------------
# Plumbing. Finished; read it once, then leave it alone.
# ---------------------------------------------------------------------------

def check_design():
    """Refuse to run against placeholder text. A gate, not a nag."""
    problems = []
    if "TODO(you)" in TASK_NAME:
        problems.append("TASK_NAME is still the placeholder")
    if "TODO(you)" in SUCCESS_CRITERION:
        problems.append("SUCCESS_CRITERION is still the placeholder")
    if any("TODO(you)" in mode for mode in FAILURE_MODES):
        problems.append("FAILURE_MODES still contains placeholders")
    if problems:
        print("Fill in the experimental design before recording trials:")
        for problem in problems:
            print(f"  - {problem}")
        print("\nWriting the success criterion after seeing the trials is how "
              "an honest person accidentally publishes a dishonest number.")
        sys.exit(1)


def load_done(path):
    """Trial indices already written, so --resume never re-rolls a trial."""
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                done.add(json.loads(line)["trial"])
    return done


def append(path, record):
    with open(path, "a") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def ask_outcome():
    """s = success, f = failure, x = void this trial (with a reason)."""
    while True:
        answer = input("  outcome [s]uccess / [f]ailure / [x] void: ").strip().lower()
        if answer in ("s", "f", "x"):
            return answer
        print("  s, f or x.")


def ask_failure_mode():
    for index, mode in enumerate(FAILURE_MODES, 1):
        print(f"    {index}. {mode}")
    while True:
        answer = input("  failure mode: ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(FAILURE_MODES):
            return FAILURE_MODES[int(answer) - 1]
        print(f"  1 to {len(FAILURE_MODES)}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--out", default="results.jsonl")
    parser.add_argument("--seed", type=int, default=0,
                        help="fixed so the condition order is reproducible")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the dealt conditions and stop")
    parser.add_argument("--notes", default="",
                        help="policy checkpoint, lighting, anything that "
                             "distinguishes this run from the next one")
    args = parser.parse_args()

    check_design()

    rng = random.Random(args.seed)
    conditions = [sample_initial_condition(rng) for _ in range(args.trials)]

    if args.dry_run:
        for index, condition in enumerate(conditions):
            print(f"trial {index + 1:2d}  {json.dumps(condition)}")
        return

    done = load_done(args.out) if args.resume else set()
    if done and not args.resume:
        sys.exit(f"{args.out} already has trials in it; pass --resume or "
                 "choose a new filename. Overwriting an evaluation is how "
                 "results get better than the robot.")

    print(f"task: {TASK_NAME}")
    print(f"success means: {SUCCESS_CRITERION}")
    print(f"logging to {args.out}. Start the camera now, and do not stop it.\n")

    for index, condition in enumerate(conditions):
        trial = index + 1
        if trial in done:
            continue
        print(f"--- trial {trial} of {args.trials}")
        print(f"  set up: {json.dumps(condition)}")
        input("  press ENTER when the table is set and the camera is rolling ")
        started = time.time()
        input("  press ENTER when the rollout has finished ")
        elapsed = time.time() - started

        outcome = ask_outcome()
        record = {
            "trial": trial,
            "condition": condition,
            "outcome": {"s": "success", "f": "failure", "x": "void"}[outcome],
            "seconds": round(elapsed, 1),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "notes": args.notes,
        }
        if outcome == "f":
            record["failure_mode"] = ask_failure_mode()
        if outcome == "x":
            record["void_reason"] = input("  why is this trial void? ").strip()
            print("  Voids are counted and printed in the report. A void is "
                  "for a broken setup, never for a failure you dislike.")
        record["comment"] = input("  one line, what you saw: ").strip()
        append(args.out, record)
        print()

    print(f"done. Now: python eval_report.py {args.out}")


if __name__ == "__main__":
    main()
