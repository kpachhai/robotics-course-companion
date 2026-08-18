"""Solution - Lesson 3.1: audit a demonstration dataset before training on it.

Run:  python dataset_audit.py                     the 50-episode SO-101 dataset
      python dataset_audit.py --path pick_demos   your own Module 2 recording
"""
import argparse

import numpy as np

from demos import DEFAULT_REPO, Demos, load


def episode_lengths(demos: Demos) -> np.ndarray:
    counted = demos.lengths()
    lengths = np.array([counted[int(e)] for e in demos.episode_ids], dtype=int)

    declared = demos.declared_lengths
    if declared:
        disagree = {e: (declared[e], counted[e]) for e in counted
                    if e in declared and declared[e] != counted[e]}
        if disagree:
            raise ValueError(
                f"metadata disagrees with the frame table for {len(disagree)} episode(s): "
                f"{dict(list(disagree.items())[:3])} (declared, counted)")
    return lengths


def joint_coverage(demos: Demos) -> np.ndarray:
    return np.stack([demos.action.min(0), demos.action.max(0)], axis=1)


def frozen_frames(demos: Demos):
    action, episode = demos.action, demos.episode

    same_as_previous = np.all(np.isclose(action[1:], action[:-1]), axis=1)
    within_episode = episode[1:] == episode[:-1]
    fraction = float((same_as_previous & within_episode).sum() / within_episode.sum())

    leading = []
    for e in demos.episode_ids:
        block = action[episode == e]
        run = 0
        while run + 1 < len(block) and np.allclose(block[run + 1], block[0]):
            run += 1
        leading.append(run)
    return fraction, np.array(leading, dtype=int)


def start_state_spread(demos: Demos) -> np.ndarray:
    firsts = np.stack([demos.state[demos.episode == e][0] for e in demos.episode_ids])
    return firsts.max(0) - firsts.min(0)


def tracking_error(demos: Demos) -> np.ndarray:
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
