"""Solution - Lesson 2.17: record episodes in the LeRobot v3.0 frame schema.

Two rates, one contract. Physics runs at the model's timestep; the controller
decides once per dataset frame and holds that command until the next one. The
action stored at frame t is therefore exactly the command that carried the robot
from the state at t to the state at t + 1, which is what replay and imitation
learning both assume.

Run:  python record_dataset.py                    3 episodes -> ./pick_demos
      python record_dataset.py -n 50 --out pick50
      python record_dataset.py --check pick_demos
      python record_dataset.py --replay            record 1, replay it, compare
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
    """The feature schema. Everything else in this file is checked against it."""
    schema = {
        "observation.state": {"dtype": "float32", "shape": [6], "names": JOINTS},
        "action": {"dtype": "float32", "shape": [6], "names": JOINTS},
    }
    for cam in cameras:
        schema[f"observation.images.{cam}"] = {
            "dtype": "video", "shape": [height, width, 3],
            "names": ["height", "width", "channels"]}
    return schema


def check_frame(frame, schema):
    """Reject a frame that would poison the dataset. Loud, at record time."""
    missing = set(schema) - set(frame)
    extra = set(frame) - set(schema) - {"task"}
    if missing or extra:
        raise ValueError(f"frame keys wrong: missing {sorted(missing)}, unexpected {sorted(extra)}")
    if not isinstance(frame.get("task"), str) or not frame["task"]:
        raise ValueError("every frame needs a non-empty 'task' string")
    for key, spec in schema.items():
        value = np.asarray(frame[key])
        want = NUMPY_DTYPE[spec["dtype"]]
        if value.dtype != want:
            raise ValueError(f"{key}: dtype is {value.dtype}, schema says {want.__name__}")
        if list(value.shape) != spec["shape"]:
            raise ValueError(f"{key}: shape is {list(value.shape)}, schema says {spec['shape']}")
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ValueError(f"{key}: contains NaN or inf")


class Recorder:
    """Frames in, a staging directory out.

    This is deliberately not the LeRobot on-disk format. It holds the frames in
    the LeRobot *schema* so that the twenty-line writer in the lesson can push
    them into a LeRobotDataset on a machine that has torch. Splitting there means
    recording needs MuJoCo and nothing else.
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
        """Write the metadata. Skip this and the directory is a pile of arrays."""
        if self.frames:
            raise ValueError("finalize with an unsaved episode still open")
        info = dict(codebase_version="v3.0", robot_type=self.robot_type, fps=self.fps,
                    total_episodes=len(self.episodes), total_frames=self.total_frames,
                    total_tasks=1, features=self.schema)
        (self.root / "info.json").write_text(json.dumps(info, indent=2))
        (self.root / "tasks.json").write_text(json.dumps({"0": self.task}, indent=2))
        with open(self.root / "episodes.jsonl", "w") as handle:
            for episode in self.episodes:
                handle.write(json.dumps(episode) + "\n")
        return info


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
        """Observe, then command, then let the world run for exactly one frame."""
        command = np.asarray(command, np.float32)      # execute what you store, exactly
        mujoco.mj_forward(model, data)                 # derived values match qpos now
        if recorder is not None:
            payload = {"observation.state": data.qpos[:6].astype(np.float32),
                       "action": command,
                       "task": recorder.task}
            if renderer is not None:
                renderer.update_scene(data, camera=CAMERA)
                payload[f"observation.images.{CAMERA}"] = renderer.render()
            recorder.add_frame(payload)
        actions.append(command.copy())
        data.ctrl[:] = command
        for _ in range(steps):
            mujoco.mj_step(model, data)

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
