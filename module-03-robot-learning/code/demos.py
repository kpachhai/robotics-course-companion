"""Read a demonstration dataset without installing lerobot.

Given to you complete. It is the machinery Lessons 3.1 and 3.2 sit on, not an
exercise itself.

A LeRobotDataset v3.0 is Parquet plus JSON in a directory. Nothing about reading
one needs the training framework, and saying so out loud is half of Lesson 3.1:
your data outlives whichever library is fashionable this quarter.

Two sources, one shape:

    from_hub("lerobot/svla_so101_pickplace")   # 50 real SO-101 episodes
    from_recording("~/robot/pick_demos")       # what Module 2's recorder wrote

Both return `Demos`, which is six numpy arrays and a few strings.

Run:  python demos.py                     summarise the Hub dataset
      python demos.py --path pick_demos   summarise your own recording
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_REPO = "lerobot/svla_so101_pickplace"


@dataclass
class Demos:
    """Every demonstration in one table, plus the labels needed to read it.

    state:    (frames, dims) what the robot reported about itself
    action:   (frames, dims) what it was told to do at that instant
    episode:  (frames,)      which demonstration each frame belongs to
    """

    state: np.ndarray
    action: np.ndarray
    episode: np.ndarray
    fps: float
    joint_names: list[str]
    task: str
    source: str
    declared_lengths: dict[int, int] | None = None  # episode lengths per metadata

    @property
    def episode_ids(self) -> np.ndarray:
        return np.unique(self.episode)

    def lengths(self) -> dict[int, int]:
        """Frames per episode, counted from the frame table itself."""
        ids, counts = np.unique(self.episode, return_counts=True)
        return {int(i): int(c) for i, c in zip(ids, counts)}

    def mask(self, episode_ids) -> np.ndarray:
        return np.isin(self.episode, np.asarray(episode_ids))

    def split(self, n_train: int):
        """First `n_train` episodes for fitting, the rest held out. Never shuffle
        frames across this line: neighbouring frames are nearly identical, so a
        frame-level split leaks the answer and every number you measure is a lie.
        """
        ids = self.episode_ids
        return ids[:n_train], ids[n_train:]

    def describe(self) -> str:
        lengths = np.array(list(self.lengths().values()))
        return (
            f"{self.source}\n"
            f"  task           {self.task!r}\n"
            f"  episodes       {len(lengths)}\n"
            f"  frames         {len(self.state)}  ({len(self.state) / self.fps / 60:.1f} min at {self.fps:g} fps)\n"
            f"  dims           state {self.state.shape[1]}, action {self.action.shape[1]}\n"
            f"  joints         {', '.join(self.joint_names)}\n"
            f"  episode length {lengths.min()}-{lengths.max()} frames "
            f"({lengths.min() / self.fps:.1f}-{lengths.max() / self.fps:.1f} s)"
        )


def from_hub(repo_id: str = DEFAULT_REPO) -> Demos:
    """Pull a LeRobotDataset v3.0 straight off the Hugging Face Hub.

    Downloads the metadata and the numeric columns only. The camera streams are
    separate MP4 files and are never touched here, which is why this costs well
    under a megabyte instead of several hundred.
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download, list_repo_files

    def grab(name):
        return hf_hub_download(repo_id, name, repo_type="dataset")

    info = json.loads(Path(grab("meta/info.json")).read_text())
    files = list_repo_files(repo_id, repo_type="dataset")

    frames = pd.concat(
        [pd.read_parquet(grab(f)) for f in sorted(f for f in files if f.startswith("data/"))],
        ignore_index=True,
    )

    declared = None
    episode_meta = sorted(f for f in files if f.startswith("meta/episodes/"))
    if episode_meta:
        meta = pd.concat([pd.read_parquet(grab(f)) for f in episode_meta], ignore_index=True)
        declared = dict(zip(meta["episode_index"].astype(int), meta["length"].astype(int)))

    task = "unknown"
    if "meta/tasks.parquet" in files:
        tasks = pd.read_parquet(grab("meta/tasks.parquet"))
        task = str(tasks.index[0]) if len(tasks.index) else "unknown"

    return Demos(
        state=np.stack(frames["observation.state"].to_numpy()).astype(np.float32),
        action=np.stack(frames["action"].to_numpy()).astype(np.float32),
        episode=frames["episode_index"].to_numpy().astype(np.int64),
        fps=float(info["fps"]),
        joint_names=list(info["features"]["action"]["names"]),
        task=task,
        source=f"hub:{repo_id}",
        declared_lengths=declared,
    )


def from_recording(root) -> Demos:
    """Read the staging directory Module 2's recorder wrote.

    Same schema as the Hub datasets - the same feature names, the same dtypes,
    one action per frame - held in .npz files instead of Parquet so that
    recording needs a physics engine and nothing else.
    """
    root = Path(root).expanduser()
    info = json.loads((root / "info.json").read_text())
    episodes = [json.loads(line) for line in (root / "episodes.jsonl").read_text().splitlines()]

    states, actions, index = [], [], []
    for episode in episodes:
        blob = np.load(root / "episodes" / f"episode-{episode['episode_index']:04d}.npz")
        states.append(blob["observation.state"].astype(np.float32))
        actions.append(blob["action"].astype(np.float32))
        index.append(np.full(len(blob["action"]), episode["episode_index"], dtype=np.int64))

    tasks = json.loads((root / "tasks.json").read_text())
    return Demos(
        state=np.concatenate(states),
        action=np.concatenate(actions),
        episode=np.concatenate(index),
        fps=float(info["fps"]),
        joint_names=list(info["features"]["action"]["names"]),
        task=str(next(iter(tasks.values()))),
        source=f"recording:{root}",
        declared_lengths={int(e["episode_index"]): int(e["length"]) for e in episodes},
    )


def load(path=None, repo_id=DEFAULT_REPO) -> Demos:
    """`--path` wins if given, otherwise the Hub."""
    return from_recording(path) if path else from_hub(repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="a directory written by Module 2's recorder")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    args = parser.parse_args()
    print(load(args.path, args.repo).describe())
