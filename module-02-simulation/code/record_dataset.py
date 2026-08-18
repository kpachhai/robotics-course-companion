"""Lesson 2.17 - record episodes in the LeRobot v3.0 frame schema.

Four things are yours, marked TODO(you). Everything else is written: the
scripted expert, the replay harness, the collection loop and the checker.

Needs your COMPLETED so101_pick.py and task_scene.xml from Lesson 2.11 in this
directory: the bin, the cube and cube_in_bin all filled in. Without them the
scene has nothing to pick up and every episode fails on the first place_cube.

Run:  python record_dataset.py --replay             TODO 3 makes this work
      python record_dataset.py -n 3                 TODO 1, 2 and 4 as well
      python record_dataset.py --check pick_demos
"""
import argparse
import json
import shutil
from pathlib import Path

import mujoco
import numpy as np

import so101_pick as sp

FPS = 25                       # 1/25 s is exactly 8 physics steps at timestep 0.005
CAMERA = "overhead"
IMG_H, IMG_W = 128, 128
TASK = "put the red cube in the bin"
ROBOT_TYPE = "so101_follower"

# LeRobot names one scalar per actuator, in actuator order. These are the SO-101
# names used by the published datasets, so a policy trained here reads the same
# columns as one trained on the real arm.
JOINTS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
          "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]

NUMPY_DTYPE = {"float32": np.float32, "int64": np.int64,
               "video": np.uint8, "image": np.uint8}


def features(cameras=(CAMERA,), height=IMG_H, width=IMG_W):
    """TODO(you) 1 - return the feature schema.

    A dict keyed by feature name. Each value is a dict with "dtype", "shape"
    and "names". You need three entries:

      observation.state          float32, shape [6], names = JOINTS
      action                     float32, shape [6], names = JOINTS
      observation.images.<cam>   dtype "video", shape [height, width, 3],
                                 names ["height", "width", "channels"]

    One image entry per camera. Nothing else: the writer adds timestamp,
    frame_index, episode_index, index and task_index itself.
    """
    raise NotImplementedError


def check_frame(frame, schema):
    """TODO(you) 2 - reject a frame that would poison the dataset.

    Raise ValueError if any of these is true, and say which:
      - the frame's keys are not exactly the schema's keys plus "task"
      - "task" is missing or is not a non-empty string
      - a value's dtype is not NUMPY_DTYPE[spec["dtype"]]
      - a value's shape is not spec["shape"]
      - a float value contains NaN or inf

    Cheap to write, and it is the difference between finding a mistake in the
    first episode and finding it after fifty.
    """
    raise NotImplementedError


class Recorder:
    """Frames in, a staging directory out.

    This is deliberately not the LeRobot on-disk format. It holds the frames in
    the LeRobot *schema* so that the writer in the lesson can push them into a
    LeRobotDataset on a machine that has torch. Splitting there means recording
    needs MuJoCo and nothing else.
    """

    def __init__(self, root, schema, fps=FPS, task=TASK, robot_type=ROBOT_TYPE):
        self.root = Path(root)
        self.schema = schema
        self.fps = fps
        self.task = task
        self.robot_type = robot_type
        self.frames = []
        self.episodes = []
        self.total_frames = 0
        if self.root.exists():
            shutil.rmtree(self.root)
        (self.root / "episodes").mkdir(parents=True)

    def add_frame(self, frame):
        check_frame(frame, self.schema)
        self.frames.append({k: np.asarray(v) for k, v in frame.items() if k != "task"})

    def save_episode(self, **meta):
        """Close the current episode. Timestamps come from the frame index, never a clock."""
        if not self.frames:
            raise ValueError("save_episode with no frames")
        index = len(self.episodes)
        columns = {key: np.stack([f[key] for f in self.frames]) for key in self.schema}
        columns["timestamp"] = (np.arange(len(self.frames)) / self.fps).astype(np.float32)
        np.savez(self.root / "episodes" / f"episode-{index:04d}.npz", **columns)
        self.episodes.append(dict(episode_index=index, length=len(self.frames),
                                  task=self.task, **meta))
        self.total_frames += len(self.frames)
        self.frames = []
        return index

    def finalize(self):
        """TODO(you) 4 - write the metadata, and refuse to do it half-done.

        Raise ValueError if self.frames is non-empty: an episode is still open
        and its rows would be silently lost.

        Then write three files under self.root and return the info dict:
          info.json      codebase_version "v3.0", robot_type, fps,
                         total_episodes, total_frames, total_tasks, features
          tasks.json     {"0": self.task}
          episodes.jsonl one JSON object per line from self.episodes

        Without this the directory is a pile of arrays with no schema. The real
        LeRobot writer has the same rule and a worse failure: its Parquet
        footers are never written, so the files do not load at all.
        """
        raise NotImplementedError


def decimation(model, fps=FPS):
    """Physics steps per dataset frame. Must be a whole number or timestamps drift."""
    ratio = 1.0 / (fps * model.opt.timestep)
    if abs(ratio - round(ratio)) > 1e-9:
        raise ValueError(
            f"fps={fps} does not divide the physics rate: 1/{fps} s is "
            f"{ratio:.4f} steps of {model.opt.timestep} s. Pick an fps whose "
            f"period is a whole number of timesteps, or change the timestep.")
    return int(round(ratio))


def waypoints(arm, cube):
    """The scripted expert's plan, as (target position, gripper command, seconds)."""
    pre = sp.radial_back(np.array([cube[0], cube[1], cube[2] + 0.07]), sp.GRASP_BACK)
    grab = sp.radial_back(np.array([cube[0], cube[1], cube[2]]), sp.GRASP_BACK)
    bin_xy = arm.model.body("bin").pos[:2]
    over = np.array([bin_xy[0], bin_xy[1], sp.CARRY_Z])
    away = np.array([bin_xy[0] * 0.6, bin_xy[1] * 0.6, 0.11])
    return [(pre, sp.OPEN, 1.0), (grab, sp.OPEN, 0.8), (grab, sp.CLOSE, 0.6),
            (pre, sp.CLOSE, 0.7), (over, sp.CLOSE, 1.2), (over, sp.OPEN, 0.4),
            (away, sp.OPEN, 0.6)]


def run_episode(arm, recorder=None, renderer=None, fps=FPS, settle=0.3, tail=1.0):
    """One scripted pick and place, driven one dataset frame at a time."""
    model, data = arm.model, arm.data
    steps = decimation(model, fps)
    placed = arm.cube()                                # before anything settles
    actions = []

    def frame(command):
        """TODO(you) 3 - one dataset frame, in the order the contract requires.

        Five things happen here, and the order is the whole lesson:

          1. cast command to float32, so the simulator executes the same number
             the dataset stores
          2. mujoco.mj_forward, so the observation matches the current qpos
          3. if recorder is not None, build the frame dict and add it:
                observation.state              data.qpos[:6] as float32
                action                         the command
                observation.images.<CAMERA>    renderer.update_scene(data,
                                                 camera=CAMERA) then .render(),
                                               only when renderer is not None
                task                           recorder.task
          4. write the command into data.ctrl
          5. mujoco.mj_step exactly `steps` times, holding that command

        Also append the command to `actions`, which is what --replay feeds back.

        Observe, then command, then step. Swap steps 3 and 5 and every row of
        the dataset holds the state that the action already produced, which is a
        policy that has seen the future.
        """
        raise NotImplementedError

    for _ in range(int(round(settle * fps))):
        frame(data.ctrl.copy())

    cube_start = arm.cube()
    q = data.ctrl[:5].copy()
    here = data.site_xpos[arm.site].copy()
    lifted = False
    for target, grip, seconds in waypoints(arm, cube_start):
        knots = 16
        path = [q.copy()]
        for k in range(1, knots + 1):
            waypoint = here + (k / knots) * (target - here)
            path.append(sp.ik(model, arm._scratch, arm.site, waypoint, path[-1])[0])
        path = np.array(path)
        grip0 = data.ctrl[5]
        count = int(round(seconds * fps))
        for i in range(count):
            s = (i + 1) / count
            s = 3 * s * s - 2 * s ** 3                 # ease in, ease out
            f = s * knots
            k0 = int(np.floor(f))
            command = np.empty(6)
            command[:5] = path[k0] + (f - k0) * (path[min(k0 + 1, knots)] - path[k0])
            command[5] = grip0 + s * (grip - grip0)
            frame(command)
        q, here = path[-1].copy(), np.asarray(target, float).copy()
        lifted = lifted or arm.cube()[2] > 0.05

    for _ in range(int(round(tail * fps))):
        frame(data.ctrl.copy())

    return dict(success=bool(sp.cube_in_bin(arm)), lifted=bool(lifted),
                frames=len(actions), actions=np.array(actions),
                cube_placed=placed.tolist(), cube_start=cube_start.tolist(),
                cube_end=arm.cube().tolist())


def replay(arm, actions, cube_placed, fps=FPS):
    """Feed the recorded actions back, holding each for one frame. Nothing else.

    Also returns the state observed at each frame, in the same order the recorder
    would have written it, so the caller can compare it against the stored
    observation column. That comparison is what catches a phase error: the
    actions alone reproduce the motion whether or not they line up with the
    observations they were stored beside.
    """
    steps = decimation(arm.model, fps)
    arm.reset()
    arm.place_cube(*cube_placed[:2], z=cube_placed[2])
    states = []
    for command in actions:
        mujoco.mj_forward(arm.model, arm.data)
        states.append(arm.data.qpos[:6].astype(np.float32))
        arm.data.ctrl[:] = command
        for _ in range(steps):
            mujoco.mj_step(arm.model, arm.data)
    return dict(success=bool(sp.cube_in_bin(arm)), cube_end=arm.cube(),
                states=np.array(states))


def collect(count, out, seed=0, half_width=0.06, images=True, fps=FPS):
    """Record `count` successful episodes. Failures are dropped, and counted."""
    arm = sp.Arm()
    renderer = mujoco.Renderer(arm.model, IMG_H, IMG_W) if images else None
    recorder = Recorder(out, features() if images else
                        {k: v for k, v in features().items() if "images" not in k}, fps=fps)
    rng = np.random.default_rng(seed)
    kept = attempts = 0
    while kept < count:
        attempts += 1
        arm.reset()
        offset = rng.uniform(-half_width, half_width, size=2)
        arm.place_cube(0.22 + offset[0], 0.0 + offset[1])
        out_ep = run_episode(arm, recorder, renderer, fps=fps)
        if out_ep["success"]:
            recorder.save_episode(seed=seed, cube_placed=out_ep["cube_placed"],
                                  cube_end=out_ep["cube_end"], attempt=attempts)
            kept += 1
        else:
            recorder.frames = []                       # drop the failed episode
        print(f"  episode {attempts:3d}: {out_ep['frames']:4d} frames  "
              f"success={out_ep['success']}  kept={kept}/{count}")
    info = recorder.finalize()
    size = sum(p.stat().st_size for p in Path(out).rglob("*"))
    print(f"\n{info['total_episodes']} episodes, {info['total_frames']} frames, "
          f"{attempts - count} discarded, {size / 1e6:.1f} MB in {out}")
    return info


def check(root, fps=FPS):
    """Reload the staging directory and prove the three things that matter."""
    root = Path(root)
    info = json.loads((root / "info.json").read_text())
    schema = info["features"]
    episodes = [json.loads(line) for line in (root / "episodes.jsonl").read_text().splitlines()]
    print(f"{info['total_episodes']} episodes, {info['total_frames']} frames, "
          f"{info['fps']} fps, features {sorted(schema)}")

    total = 0
    for episode in episodes:
        blob = np.load(root / "episodes" / f"episode-{episode['episode_index']:04d}.npz")
        for key, spec in schema.items():
            column = blob[key]
            assert column.dtype == NUMPY_DTYPE[spec["dtype"]], f"{key} dtype"
            assert list(column.shape[1:]) == spec["shape"], f"{key} shape"
            assert len(column) == episode["length"], f"{key} length"
        gap = np.diff(blob["timestamp"])
        assert np.allclose(gap, 1 / info["fps"], atol=1e-6), "timestamps are not on the grid"
        total += episode["length"]
    assert total == info["total_frames"], "episode lengths do not sum to total_frames"
    print(f"schema, lengths and timestamps check out ({total} frames)")

    arm = sp.Arm()
    for episode in episodes[:3]:
        blob = np.load(root / "episodes" / f"episode-{episode['episode_index']:04d}.npz")
        out = replay(arm, blob["action"], np.array(episode["cube_placed"]), fps=fps)
        drift = np.linalg.norm(out["cube_end"] - np.array(episode["cube_end"]))
        phase = np.abs(out["states"] - blob["observation.state"]).max()
        print(f"  episode {episode['episode_index']}: replay success={out['success']}  "
              f"cube lands {drift * 1000:.3f} mm away  "
              f"observations differ by at most {np.degrees(phase):.6f} deg")
        assert out["success"], "the recorded actions do not reproduce the recorded outcome"
        assert phase == 0.0, "observation column is out of phase with the action column"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=3, help="episodes to keep")
    parser.add_argument("--out", default="pick_demos")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=float, default=0.06, help="cube half-width, metres")
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--check", metavar="DIR", help="verify an existing dataset")
    parser.add_argument("--replay", action="store_true", help="record one, replay it, compare")
    args = parser.parse_args()

    if args.check:
        check(args.check)
        return
    if args.replay:
        arm = sp.Arm()
        arm.place_cube(0.22, 0.0)
        first = run_episode(arm)
        second = replay(arm, first["actions"], np.array(first["cube_placed"]))
        print(f"recorded: success={first['success']} cube={np.round(first['cube_end'], 5)}")
        print(f"replayed: success={second['success']} cube={np.round(second['cube_end'], 5)}")
        print(f"identical: {np.array_equal(np.array(first['cube_end']), second['cube_end'])}")
        return
    collect(args.n, args.out, seed=args.seed, half_width=args.width, images=not args.no_images)


if __name__ == "__main__":
    main()
