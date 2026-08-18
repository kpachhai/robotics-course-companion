"""Lesson 2.3 - prove the MuJoCo install works, in one file.

Run:  python verify_mujoco.py        (no mjpython needed - nothing here opens a window)

Four TODOs. When they all pass you have a working simulator, a physics claim you
checked against arithmetic you already trust, and a rendered frame on disk.
"""
import platform
import sys
from pathlib import Path

import mujoco
import numpy as np

BALL_XML = """
<mujoco model="hello-mujoco">
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 0.05" rgba="0.92 0.90 0.85 1"/>
    <body name="ball" pos="0 0 1.0">
      <freejoint/>
      <geom name="ball" type="sphere" size="0.05" rgba="0.18 0.36 0.49 1"/>
    </body>
  </worldbody>
</mujoco>
"""

DROP_HEIGHT = 1.0
GRAVITY = 9.81


def env_report():
    """Print the python version, the CPU architecture, and the MuJoCo version.

    mujoco.__version__ is the Python package. mujoco.mj_versionString() is the native
    engine underneath it. If those two ever disagree you have a broken install.
    """
    # TODO(you)
    raise NotImplementedError


def build():
    """Compile BALL_XML into a model, allocate its data, print the sizes, return both.

    mujoco.MjModel.from_xml_string(BALL_XML) -> model   (the compiled, constant robot)
    mujoco.MjData(model)                     -> data    (everything that changes)
    Worth printing: model.nbody, model.nq, model.nv, model.opt.timestep.
    """
    # TODO(you)
    raise NotImplementedError


def free_fall_error(model, data, seconds=0.3):
    """Step until data.time >= seconds, then compare height against high-school physics.

    Call mujoco.mj_step(model, data) in a loop. The ball's height is data.qpos[2].
    The textbook answer is DROP_HEIGHT - 0.5 * GRAVITY * data.time**2.
    Print both and return the gap in millimetres.

    Expect a gap of a few millimetres, not zero. The simulator adds up small steps
    instead of solving the equation; Lesson 2.2 is about exactly that difference.
    """
    # TODO(you)
    raise NotImplementedError


def settle(model, data, seconds=3.0):
    """Keep stepping until the ball is lying still on the floor."""
    while data.time < seconds:
        mujoco.mj_step(model, data)
    radius = float(model.geom("ball").size[0])
    print(f"at rest: z={float(data.qpos[2]):.5f} m, ball radius {radius} m, "
          f"{data.ncon} contact(s)")


def render_frame(model, data, path=None, width=480, height=360):
    """Render one frame offscreen and save it. On macOS this goes through CGL, no window.

        import matplotlib.image as mpimg
        with mujoco.Renderer(model, height, width) as renderer:
            renderer.update_scene(data)
            pixels = renderer.render()          # (height, width, 3) uint8
        mpimg.imsave(path, pixels)

    Note the argument order: Renderer takes height first, then width.

    `path` is resolved beside this script rather than in the current directory:
    run from the repo root, a bare filename would drop an untracked PNG there,
    and only `module-*/code/*.png` is gitignored.
    """
    path = Path(__file__).with_name("ball.png") if path is None else Path(path)
    # TODO(you)
    raise NotImplementedError


if __name__ == "__main__":
    env_report()
    model, data = build()
    gap = free_fall_error(model, data)
    assert gap < 10.0, "free fall is off by more than a centimetre; something is very wrong"
    settle(model, data)
    render_frame(model, data)
    print("\nMuJoCo works. Nothing above opened a window, so all of it runs under plain python.")
