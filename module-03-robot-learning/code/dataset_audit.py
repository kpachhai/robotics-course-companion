"""Lesson 3.1 - audit a demonstration dataset before you train anything on it.

Five questions, asked of any LeRobot-schema dataset:

  1. how long is an episode, and how much do the lengths disagree
  2. how much of each joint's range did the demonstrations ever visit
  3. how many frames carry no new command at all
  4. how alike are the starting poses
  5. how far behind the command did the arm actually run

Three of them are yours to write. The other two are done, as a worked pattern.

Run:  python dataset_audit.py                     the 50-episode SO-101 dataset
      python dataset_audit.py --path pick_demos   your own Module 2 recording
"""
import argparse

import numpy as np

from demos import DEFAULT_REPO, Demos, load


def episode_lengths(demos: Demos) -> np.ndarray:
    """Frames in each episode, ordered by episode index.

    Also cross-check against `demos.declared_lengths` when the dataset carries
    metadata, and raise if the two disagree. A length column that does not match
    the rows it describes is the classic silent-corruption bug: nothing fails at
    write time, and the loader hands you a torn episode months later.
    """
    # TODO(you): return the counted lengths as an int array, and raise
    # ValueError if declared_lengths exists and disagrees with the count.
    raise NotImplementedError


def joint_coverage(demos: Demos) -> np.ndarray:
    """(dims, 2) array of [min, max] of the commanded action, per joint.

    This is the honest statement of what the dataset taught. A policy fitted to
    it has seen nothing outside these intervals and has no reason to behave
    sensibly outside them.
    """
    # TODO(you): one line with np.min / np.max over the frame axis.
    raise NotImplementedError


def frozen_frames(demos: Demos):
    """(fraction, leading) - how much of the dataset carries no new command.

    fraction: share of frames whose action is identical to the previous frame's,
              counted within an episode so the boundary is not a false positive.
    leading:  per episode, how many frames at the start repeat the first action.
              That is dead air before the demonstration begins.
    """
    # TODO(you): compute both. Watch the episode boundaries.
    raise NotImplementedError


def start_state_spread(demos: Demos) -> np.ndarray:
    """Per joint, the spread of the first observed state across episodes.

    Given complete. A tight spread here means every demonstration began from
    nearly the same pose, so the policy will only ever have seen one way in.
    """
    firsts = np.stack([demos.state[demos.episode == e][0] for e in demos.episode_ids])
    return firsts.max(0) - firsts.min(0)


def tracking_error(demos: Demos) -> np.ndarray:
    """Mean absolute gap between the command and the state, per joint.

    Given complete. The action is what the arm was *told*; the state is where it
    actually was. On a teleoperated arm the follower lags the leader, so these
    two columns are never equal, and confusing them is a real bug you can ship.
    """
    return np.abs(demos.action - demos.state).mean(0)


def report(demos: Demos) -> None:
    print(demos.describe())
    fps, names = demos.fps, demos.joint_names

    lengths = episode_lengths(demos)
    print(f"\nepisode length      median {np.median(lengths):.0f} frames "
          f"({np.median(lengths) / fps:.1f} s), "
          f"longest / shortest = {lengths.max() / lengths.min():.2f}x")

    coverage = joint_coverage(demos)
    print("\njoint coverage (commanded action)")
    for name, (low, high) in zip(names, coverage):
        bar = "#" * max(1, int(round((high - low) / 200 * 40)))
        print(f"  {name:<18} {low:8.2f} .. {high:8.2f}   span {high - low:7.2f}  {bar}")

    fraction, leading = frozen_frames(demos)
    print(f"\nframes with no new command   {fraction * 100:.1f}%")
    print(f"dead air at episode start    median {np.median(leading):.0f} frames "
          f"({np.median(leading) / fps:.2f} s), worst {leading.max()} "
          f"({leading.max() / fps:.2f} s)")

    print("\nstart-state spread across episodes (max - min)")
    for name, spread in zip(names, start_state_spread(demos)):
        print(f"  {name:<18} {spread:7.2f}")

    print("\nmean |action - state| per joint")
    for name, gap in zip(names, tracking_error(demos)):
        print(f"  {name:<18} {gap:7.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="a directory written by Module 2's recorder")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    args = parser.parse_args()
    report(load(args.path, args.repo))
