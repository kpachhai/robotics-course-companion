"""Solution - Lesson 2.3. Prove the MuJoCo install works, in one file.

Run:  python verify_mujoco.py        (no mjpython needed - nothing here opens a window)
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
    print(f"python  {sys.version.split()[0]}  {platform.machine()}  {platform.system()}")
    print(f"mujoco  {mujoco.__version__}  (engine reports {mujoco.mj_versionString()})")


def build():
    model = mujoco.MjModel.from_xml_string(BALL_XML)
    data = mujoco.MjData(model)
    print(f"compiled: {model.nbody} bodies, nq={model.nq}, nv={model.nv}, "
          f"timestep={model.opt.timestep} s")
    return model, data


def free_fall_error(model, data, seconds=0.3):
    """Step until `seconds`, then compare height against high-school physics."""
    while data.time < seconds:
        mujoco.mj_step(model, data)
    simulated = float(data.qpos[2])
    exact = DROP_HEIGHT - 0.5 * GRAVITY * data.time ** 2
    gap_mm = abs(simulated - exact) * 1000
    print(f"t={data.time:.3f} s   simulated z={simulated:.5f} m   "
          f"textbook z={exact:.5f} m   gap={gap_mm:.2f} mm")
    return gap_mm


def settle(model, data, seconds=3.0):
    """Keep stepping until the ball is lying still on the floor."""
    while data.time < seconds:
        mujoco.mj_step(model, data)
    radius = float(model.geom("ball").size[0])
    print(f"at rest: z={float(data.qpos[2]):.5f} m, ball radius {radius} m, "
          f"{data.ncon} contact(s)")


def render_frame(model, data, path=None, width=480, height=360):
    """Offscreen render. On macOS this goes through CGL and needs no window.

    `path` defaults to a file beside this script rather than the current
    directory: run from the repo root, a bare filename would drop an untracked
    PNG there, and only `module-*/solutions/*.png` is gitignored.
    """
    import matplotlib.image as mpimg

    path = Path(__file__).with_name("ball.png") if path is None else Path(path)
    with mujoco.Renderer(model, height, width) as renderer:
        renderer.update_scene(data)
        pixels = renderer.render()
    mpimg.imsave(path, pixels)
    print(f"wrote {path.name}  {pixels.shape} {pixels.dtype}  "
          f"mean brightness {pixels.mean():.1f}")


if __name__ == "__main__":
    env_report()
    model, data = build()
    gap = free_fall_error(model, data)
    assert gap < 10.0, "free fall is off by more than a centimetre; something is very wrong"
    settle(model, data)
    render_frame(model, data)
    print("\nMuJoCo works. Nothing above opened a window, so all of it runs under plain python.")
