"""Turn a task specification into a recording schedule and a lerobot-record command.

HONESTY NOTE
------------
This script never touches hardware. It does arithmetic on the numbers you give
it and prints a plan and a command string. Nothing here has been validated
against a physical SO-101, because the machine this course was written on does
not have one. The command it prints uses flag names read out of LeRobot v0.6.1's
argument parsers; check them against `lerobot-record --help` on your own install
before you trust the paste.

Usage
-----
    python recording_plan.py
    python recording_plan.py --episodes 60 --cells A1 A2 A3 B1 B2 B3
    python recording_plan.py --task "put the red brick in the bowl" --repo-id me/brick_v1

Standard library only.
"""

from __future__ import annotations

import argparse
import math

DEFAULT_CELLS = ["A1", "A2", "A3", "B1", "B2", "B3"]


def episodes_per_cell(total_episodes: int, cells: list[str]) -> dict[str, int]:
    """Split episodes across cells as evenly as possible.

    The remainder is handed out one per cell from the start of the list, so the
    largest imbalance between any two cells is a single episode.
    """
    if total_episodes <= 0:
        raise ValueError("total_episodes must be positive")
    if not cells:
        raise ValueError("need at least one start cell")

    base, remainder = divmod(total_episodes, len(cells))
    return {cell: base + (1 if index < remainder else 0) for index, cell in enumerate(cells)}


def interleaved_order(per_cell: dict[str, int]) -> list[str]:
    """Order the episodes so cells alternate instead of running in blocks.

    Recording all of A1 first and all of B3 last bakes your own fatigue, and any
    drift in the light, into a position-dependent pattern the policy can learn.
    Round-robin spreads both across the whole session.
    """
    remaining = dict(per_cell)
    order: list[str] = []
    while any(count > 0 for count in remaining.values()):
        for cell in per_cell:
            if remaining[cell] > 0:
                order.append(cell)
                remaining[cell] -= 1
    return order


def session_minutes(episodes: int, episode_seconds: float, reset_seconds: float) -> float:
    """Wall clock if every episode is a keeper and nothing ever goes wrong."""
    return episodes * (episode_seconds + reset_seconds) / 60.0


def warnings_for(args: argparse.Namespace) -> list[str]:
    """Checks worth failing loudly on before two hours disappear."""
    found = []
    if args.episodes < 50:
        found.append(
            f"{args.episodes} episodes is below the 50 the LeRobot docs suggest as a floor."
        )
    if args.episode_seconds < 20 or args.episode_seconds > 45:
        found.append(
            f"episode_time_s={args.episode_seconds:g} sits outside the documented 20-45 s band."
        )
    if args.episodes % len(args.cells) != 0:
        found.append(
            f"{args.episodes} episodes over {len(args.cells)} cells does not divide evenly, "
            "so some cells get one extra."
        )
    if args.fps != 30:
        found.append(f"fps={args.fps} is not the 30 that the docs and examples assume.")
    return found


def record_command(args: argparse.Namespace) -> str:
    cameras = (
        "{ front: {type: opencv, index_or_path: %s, width: 640, height: 480, fps: %d}, "
        "wrist: {type: opencv, index_or_path: %s, width: 640, height: 480, fps: %d} }"
        % (args.front_camera, args.fps, args.wrist_camera, args.fps)
    )
    return "\n".join(
        [
            "lerobot-record \\",
            f"    --robot.type=so101_follower --robot.port={args.follower_port} "
            f"--robot.id={args.follower_id} \\",
            f'    --robot.cameras="{cameras}" \\',
            f"    --teleop.type=so101_leader --teleop.port={args.leader_port} "
            f"--teleop.id={args.leader_id} \\",
            f"    --dataset.repo_id={args.repo_id} \\",
            f"    --dataset.num_episodes={args.episodes} \\",
            f"    --dataset.episode_time_s={args.episode_seconds:g} \\",
            f"    --dataset.reset_time_s={args.reset_seconds:g} \\",
            f'    --dataset.single_task="{args.task}" \\',
            "    --dataset.streaming_encoding=true \\",
            # Without this, lerobot-record appends a _YYYYMMDD_HHMMSS suffix to
            # repo_id at creation time, and every later command that takes the
            # repo id back (replay, train, the visualiser) needs the suffixed
            # name. The flag arrived in v0.6.1; on v0.6.0 drop this line and read
            # the real name off the log instead.
            "    --dataset.no_stamp=true \\",
            "    --display_data=true",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--cells", nargs="+", default=DEFAULT_CELLS)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--episode-seconds", type=float, default=30.0)
    parser.add_argument("--reset-seconds", type=float, default=10.0)
    parser.add_argument("--task", default="put the red brick in the bowl")
    parser.add_argument("--repo-id", default="${HF_USER}/so101_brick_v1")
    parser.add_argument("--follower-port", default="/dev/ttyACM0")
    parser.add_argument("--leader-port", default="/dev/ttyACM1")
    parser.add_argument("--follower-id", default="my_follower")
    parser.add_argument("--leader-id", default="my_leader")
    parser.add_argument("--front-camera", default="/dev/video0")
    parser.add_argument("--wrist-camera", default="/dev/video2")
    args = parser.parse_args()

    per_cell = episodes_per_cell(args.episodes, args.cells)
    order = interleaved_order(per_cell)
    nominal = session_minutes(args.episodes, args.episode_seconds, args.reset_seconds)
    frames = args.episodes * args.fps * args.episode_seconds

    print(f'task: "{args.task}"')
    print(f"episodes: {args.episodes} across {len(args.cells)} start cells\n")

    print("episodes per cell")
    for cell, count in per_cell.items():
        print(f"  {cell:>4}  {count:>3}  {'#' * count}")

    print("\nrecording order (first 18 shown)")
    print("  " + " ".join(order[:18]) + (" ..." if len(order) > 18 else ""))

    print(f"\nframes if every episode is kept: {int(frames):,}")
    print(f"nominal wall clock:              {nominal:.0f} min")
    print(
        f"budget for the session:          {nominal * 3 / 60:.1f} to {nominal * 4 / 60:.1f} h"
    )
    print("  (resets, retries and discards are reported to cost 3-4x the arithmetic;")
    print("   that multiplier comes from practitioner reports, not from a measurement here)")

    problems = warnings_for(args)
    if problems:
        print("\ncheck these before you start")
        for problem in problems:
            print(f"  ! {problem}")

    print("\ncommand")
    print(record_command(args))

    print("\nkeep this plan open while you record, and mark cells off as you go.")
    print(f"if you stop early you will need {math.ceil(args.episodes / len(args.cells))} "
          "per cell at minimum for even coverage.")


if __name__ == "__main__":
    main()
