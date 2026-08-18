"""Lesson 2.6 - the simulator as a state machine: mjModel, mjData, mj_step.

Every mode below is one claim from the lesson, made runnable.

Run:  python step_loop.py                (all of them)
      python step_loop.py --gravity      "doing nothing" is still a command
      python step_loop.py --hold         data.ctrl is the only wire in
      python step_loop.py --state        nq is not nv
      python step_loop.py --forward      derived values are stale until you ask
      python step_loop.py --replay       same state in, same state out
      python step_loop.py --speed        sim seconds per wall-clock second
      python step_loop.py --menagerie ~/somewhere/else/mujoco_menagerie
"""
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

DEG = 180.0 / np.pi
ARM = "robotstudio_so101/scene.xml"
BOX = "robotstudio_so101/scene_box.xml"


def find_scene(argv, rel=ARM):
    """Locate a Menagerie scene. --menagerie beats $MENAGERIE beats ~/mujoco_menagerie."""
    root = Path.home() / "mujoco_menagerie"
    if os.environ.get("MENAGERIE"):
        root = Path(os.environ["MENAGERIE"])
    if "--menagerie" in argv:
        root = Path(argv[argv.index("--menagerie") + 1])
    path = root.expanduser() / rel
    if not path.exists():
        sys.exit(f"model not found: {path}\nclone it: git clone https://github.com/google-deepmind/mujoco_menagerie {root}")
    return path


def load(argv, rel=ARM):
    model = mujoco.MjModel.from_xml_path(str(find_scene(argv, rel)))
    return model, mujoco.MjData(model)


def sag(model, seconds=4.0, sample=1.0):
    """Advance an untouched MjData and record shoulder_lift over time."""
    data = mujoco.MjData(model)
    every = int(round(sample / model.opt.timestep))
    trace = []
    for k in range(int(seconds / model.opt.timestep) + 1):
        if k % every == 0:
            trace.append((data.time, data.qpos[1] * DEG))
        mujoco.mj_step(model, data)
    return trace


def demo_gravity(argv):
    """Leaving data.ctrl at zero is not "no command". It is the command "go to zero"."""
    model, _ = load(argv)
    print(f"\n--gravity  timestep {model.opt.timestep} s, gravity {model.opt.gravity}")
    no_servos = int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    on = sag(model)
    model.opt.disableflags |= no_servos
    off = sag(model)
    model.opt.disableflags &= ~no_servos
    print(f"  {'sim time':>9}{'shoulder_lift, servos on':>26}{'servos off':>14}")
    for (t, a), (_, b) in zip(on, off):
        print(f"  {t:8.2f}s{a:24.3f}°{b:13.2f}°")
    print("  Same model, same starting state, same number of steps.")
    print("  The only difference is one bit in model.opt.disableflags.")


def demo_hold(argv):
    """data.ctrl is the entire control surface: six numbers, written every step."""
    model, data = load(argv)
    target = np.array([0.0, -0.6, 1.2, 0.5, 0.0, 0.6])
    print("\n--hold   commanding a pose through data.ctrl")
    print(f"  target  {np.round(target * DEG, 1)}")
    for k in range(401):
        data.ctrl[:] = target
        if k % 100 == 0:
            err = np.abs(data.qpos[:6] - target).max() * DEG
            print(f"  t={data.time:5.2f}s  worst joint error {err:7.2f}°")
        mujoco.mj_step(model, data)
    print(f"  settled at {np.round(data.qpos[:6] * DEG, 1)}")
    print("  Residual error is gravity: a position servo trades a little droop for holding force.")


def demo_state(argv):
    """nq counts position numbers, nv counts velocity numbers. They differ."""
    model, data = load(argv, BOX)
    print("\n--state  scene_box.xml: the arm plus one free-floating cube")
    print(f"  nq = {model.nq}   nv = {model.nv}   nu = {model.nu}   nbody = {model.nbody}")
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "<the cube's freejoint>"
        kind = mujoco.mjtJoint(model.jnt_type[j]).name.replace("mjJNT_", "").lower()
        qadr, vadr = model.jnt_qposadr[j], model.jnt_dofadr[j]
        nq = model.nq - qadr if j == model.njnt - 1 else model.jnt_qposadr[j + 1] - qadr
        nv = model.nv - vadr if j == model.njnt - 1 else model.jnt_dofadr[j + 1] - vadr
        print(f"  {name:<24}{kind:<6} qpos[{qadr}:{qadr + nq}] ({nq})   qvel[{vadr}:{vadr + nv}] ({nv})")
    print("  The free joint stores 3 positions plus a 4-number quaternion, but only")
    print("  6 velocities. A quaternion has one redundant number, so nq > nv.")
    mujoco.mj_resetDataKeyframe(model, data, 0)
    print(f"  after mj_resetDataKeyframe(..., 0): data.time = {data.time:.3f} s")
    print("  A keyframe restores time too. Zero it yourself if you are logging episodes.")


def demo_forward(argv):
    """qpos is input. xpos is derived, and derived values are recomputed on demand."""
    model, data = load(argv)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    print("\n--forward  writing qpos does not move anything by itself")
    print(f"  fresh MjData, fingertip at {np.round(data.site_xpos[site], 4)}  (never computed)")
    mujoco.mj_forward(model, data)
    home = data.site_xpos[site].copy()
    print(f"  after mj_forward           {np.round(home, 4)}")
    data.qpos[1] = 0.8
    print(f"  after qpos[1] = 0.8        {np.round(data.site_xpos[site], 4)}  (unchanged: stale)")
    mujoco.mj_forward(model, data)
    print(f"  after mj_forward           {np.round(data.site_xpos[site], 4)}  (moved {np.linalg.norm(data.site_xpos[site] - home) * 1000:.0f} mm)")
    print(f"  data.time is still {data.time:.3f}: mj_forward computes, mj_step computes and advances.")


def demo_replay(argv):
    """Same model, same state, same inputs, same result. Bit for bit."""
    model, data = load(argv, BOX)

    def run(nsteps=600):
        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.time = 0.0
        for _ in range(nsteps):
            data.ctrl[:] = [0.0, 0.3, 0.4, 1.0, 0.0, 0.2]
            mujoco.mj_step(model, data)
        return data.qpos.copy()

    a, b = run(), run()
    print("\n--replay  two runs from the same keyframe")
    print(f"  cube position run 1: {np.round(a[6:9], 12)}")
    print(f"  cube position run 2: {np.round(b[6:9], 12)}")
    print(f"  bit-identical across all {model.nq} numbers: {np.array_equal(a, b)}")
    print(f"  largest difference: {np.abs(a - b).max():.1e}")


def demo_speed(argv):
    """The reason simulation is worth the trouble."""
    model, data = load(argv, BOX)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    nsteps = 20_000
    t0 = time.perf_counter()
    mujoco.mj_step(model, data, nstep=nsteps)
    wall = time.perf_counter() - t0
    simulated = nsteps * model.opt.timestep
    print("\n--speed  how fast this machine runs one arm")
    print(f"  {nsteps} steps at {model.opt.timestep} s = {simulated:.0f} s of simulated time")
    print(f"  wall clock: {wall:.2f} s")
    print(f"  {simulated / wall:.0f}x real time, on one CPU core, with contact enabled.")


DEMOS = {
    "--gravity": demo_gravity,
    "--hold": demo_hold,
    "--state": demo_state,
    "--forward": demo_forward,
    "--replay": demo_replay,
    "--speed": demo_speed,
}


def main(argv):
    chosen = [f for f in DEMOS if f in argv] or list(DEMOS)
    for flag in chosen:
        DEMOS[flag](argv)


if __name__ == "__main__":
    main(sys.argv[1:])
