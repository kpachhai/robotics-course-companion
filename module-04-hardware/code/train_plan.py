"""Work out how long to train on your own dataset, and print the command.

HONESTY NOTE
------------
Arithmetic only. This script has never launched a training run on this machine
and cannot tell you how long yours will take in wall-clock minutes; that depends
on your GPU, your dataloader and your batch size. What it can do is stop you
passing a step count that means five hundred epochs or half an epoch.

The learning-rate scheduler flag is emitted only for the policies whose config
actually declares it. Checked against LeRobot 0.6.1: ACT has no scheduler at all
(`get_scheduler_preset()` returns None) and Diffusion Policy's scheduler exposes
`scheduler_name`/`scheduler_warmup_steps` but no decay-steps field. Passing an
undeclared nested flag does not no-op - draccus raises a DecodingError saying the
field is not valid for the chosen policy config - so emitting it unconditionally
would hand you a command that cannot run.

Flag names read out of LeRobot v0.6.1's argument parsers. Check them against
`lerobot-train --help`.

One thing this script cannot know: `--dataset.repo_id` must be the name
lerobot-record actually created, which carries a _YYYYMMDD_HHMMSS suffix unless
you recorded with `--dataset.no_stamp=true` (v0.6.1 and newer).

Usage
-----
    python train_plan.py
    python train_plan.py --episodes 72 --seconds 25 --batch 16 --epochs 8
    python train_plan.py --policy diffusion --device cuda

Standard library only.
"""

from __future__ import annotations

import argparse
import math

# Documented starting bands, from the LeRobot agent guide. These are ranges to
# aim at, not guarantees; your own success rate is the only real evidence.
POLICY_BANDS = {
    "act": (30_000, 80_000),
    "diffusion": (80_000, 150_000),
    "smolvla": (30_000, 80_000),
}

# Policies whose config declares `scheduler_decay_steps`. Anything not listed here
# rejects the flag outright. Read off the configuration_*.py files at LeRobot
# 0.6.1; newer releases add policies, so re-check if you train something exotic.
POLICIES_WITH_DECAY_STEPS = {
    "smolvla",
    "pi0",
    "pi0_fast",
    "pi05",
    "xvla",
    "eo1",
    "molmoact2",
    "wall_x",
    "vla_jepa",
}


def total_frames(episodes: int, fps: int, seconds: float) -> int:
    return int(episodes * fps * seconds)


def steps_per_epoch(frames: int, batch: int) -> int:
    return math.ceil(frames / batch)


def train_command(args: argparse.Namespace, steps: int) -> str:
    lines = [
        "lerobot-train \\",
        f"    --dataset.repo_id={args.repo_id} \\",
        f"    --policy.type={args.policy} \\",
        f"    --output_dir=outputs/train/{args.job_name} \\",
        f"    --job_name={args.job_name} \\",
        f"    --policy.device={args.device} \\",
        f"    --batch_size={args.batch} \\",
        f"    --steps={steps} \\",
    ]
    if args.policy in POLICIES_WITH_DECAY_STEPS:
        lines.append(f"    --policy.scheduler_decay_steps={steps} \\")
    lines += [
        f"    --save_freq={args.save_freq} \\",
        "    --wandb.enable=true \\",
        f"    --policy.repo_id={args.repo_id}_{args.policy}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=30.0,
                        help="average length of a kept episode, in seconds")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--policy", default="act", choices=sorted(POLICY_BANDS))
    parser.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    parser.add_argument("--repo-id", default="${HF_USER}/so101_brick_v1")
    parser.add_argument("--job-name", default="act_brick_v1")
    parser.add_argument("--save-freq", type=int, default=None,
                        help="defaults to one checkpoint every fifth of the run")
    args = parser.parse_args()

    frames = total_frames(args.episodes, args.fps, args.seconds)
    per_epoch = steps_per_epoch(frames, args.batch)
    steps = int(round(per_epoch * args.epochs))
    args.save_freq = args.save_freq or max(1_000, steps // 5)

    print(f"dataset        {args.episodes} episodes x {args.fps} fps x {args.seconds:g} s")
    print(f"frames         {frames:,}")
    print(f"batch          {args.batch}")
    print(f"steps / epoch  {per_epoch:,}")
    print(f"epochs         {args.epochs:g}")
    print(f"total steps    {steps:,}")
    print(f"checkpoints    every {args.save_freq:,} steps "
          f"({steps // args.save_freq} of them, plus the last)")

    low, high = POLICY_BANDS[args.policy]
    print(f"\ndocumented starting band for {args.policy}: {low:,} to {high:,} steps")
    if steps < low:
        print(f"  ! {steps:,} is short. Either record more episodes or raise --epochs;")
        print("    a short run under-trains rather than saving you time.")
    elif steps > high:
        print(f"  ! {steps:,} is long for this dataset. More steps on the same fifty")
        print("    episodes buys memorisation, not generalisation.")
    else:
        print("  this run sits inside the band.")

    if args.device == "cpu":
        print("\n! --policy.device=cpu will not finish in a useful time. Use a rented GPU,")
        print("  a Colab runtime, or --job.target on Hugging Face Jobs instead.")

    if args.policy in POLICIES_WITH_DECAY_STEPS:
        print("\nthis policy carries a cosine decay sized for 30,000 steps, so the command")
        print("below pins scheduler_decay_steps to your own step count. Current LeRobot")
        print("also rescales it automatically; setting it is belt-and-braces.")
    else:
        print(f"\n{args.policy} does not declare scheduler_decay_steps, so the command below")
        print("omits it. Passing it anyway aborts the run with a draccus DecodingError")
        print("rather than being ignored, which is how that widely-copied flag wastes")
        print("an evening.")

    print("\ncommand")
    print(train_command(args, steps))


if __name__ == "__main__":
    main()
