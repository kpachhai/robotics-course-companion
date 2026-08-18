"""Lesson 2.6 - the simulator as a state machine: mjModel, mjData, mj_step.

Four TODOs. Each one turns a sentence from the lesson into something you can
watch happen. --hold and --speed are done for you as worked examples of the
two shapes you will write all module: write ctrl then step, and step in bulk.

Run:  python step_loop.py                (all of them)
      python step_loop.py --gravity      "doing nothing" is still a command
      python step_loop.py --hold         data.ctrl is the only wire in
      python step_loop.py --state        nq is not nv
      python step_loop.py --forward      derived values are stale until you ask
      python step_loop.py --replay       same state in, same state out
      python step_loop.py --speed        sim seconds per wall-clock second
      python step_loop.py --menagerie ~/somewhere/else/mujoco_menagerie

The API you need, all of it:
  mujoco.MjModel.from_xml_path(path)      constants, compiled from the XML
  mujoco.MjData(model)                    state, one per simulated world
  mujoco.mj_step(model, data)             advance by model.opt.timestep
  mujoco.mj_step(model, data, nstep=N)    advance N times without returning
  mujoco.mj_forward(model, data)          recompute derived values, no advance
  mujoco.mj_resetData(model, data)        back to the model defaults
  mujoco.mj_resetDataKeyframe(model, data, k)   back to keyframe k
  data.qpos / qvel / ctrl / time / xpos / site_xpos
  model.opt.timestep / gravity / disableflags
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


def demo_gravity(argv):
    """Leaving data.ctrl at zero is not "no command". It is the command "go to zero".

    Run the arm for four seconds from a fresh MjData without ever touching
    data.ctrl, sampling shoulder_lift once a second. Then set

        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)

    and run exactly the same four seconds again. Print the two columns side by
    side. One bit of the model should be the only difference between an arm
    that holds itself up and an arm that does not.
    """
    # TODO(you)
    raise NotImplementedError


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
    """nq counts position numbers, nv counts velocity numbers. They differ.

    Load scene_box.xml, which adds one free-floating cube to the arm. Print nq,
    nv, nu and nbody, then one line per joint showing the slice of data.qpos and
    the slice of data.qvel that joint owns. Use model.jnt_qposadr and
    model.jnt_dofadr; a joint's slice runs to the next joint's address, and the
    last one runs to nq or nv. The cube's joint has no name, so print something
    sensible rather than None.

    Then call mujoco.mj_resetDataKeyframe(model, data, 0) and print data.time.
    It is not zero, and finding that out here is cheaper than finding it out in
    Lesson 2.17 with a mislabelled dataset.
    """
    # TODO(you)
    raise NotImplementedError


def demo_forward(argv):
    """qpos is input. xpos is derived, and derived values are recomputed on demand.

    On a fresh MjData, print data.site_xpos for the site named "gripperframe"
    before calling anything. Then mj_forward and print it again. Then write
    data.qpos[1] = 0.8 and print it a third time without calling anything. Then
    mj_forward and print it a fourth. Two of those four prints are equal, and
    the pair that surprises you is the lesson.
    """
    # TODO(you)
    raise NotImplementedError


def demo_replay(argv):
    """Same model, same state, same inputs, same result. Bit for bit.

    Write a local run(nsteps) that resets to keyframe 0, zeroes data.time,
    then steps nsteps times writing the same data.ctrl each step, and returns
    data.qpos.copy(). Call it twice and compare with np.array_equal - not
    np.allclose. If a floating-point physics engine is worth trusting as a test
    fixture, the comparison has to be exact.
    """
    # TODO(you)
    raise NotImplementedError


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
