"""Solution - Lesson 2.10: keyboard teleoperation of the SO-101 in MuJoCo.

Keys move a target point in world coordinates. Every physics step, one
damped-least-squares IK update pushes the commanded joint angles toward that
target. The arm chases; you steer the point.

    I / K   target forward / back   (+x / -x)
    J / L   target left / right     (+y / -y)
    U / O   target up / down        (+z / -z)
    G / H   gripper close / open
    R       target back to the tool tip's rest position

Run:  mjpython teleop_keyboard.py           (interactive, macOS)
      python   teleop_keyboard.py           (interactive, Linux and Windows)
      python   teleop_keyboard.py --replay  (scripted, no window, any platform)
"""
import os
import sys
import time

import numpy as np
import mujoco

from fk_so101 import MENAGERIE, chain_geometry, fk
from ik_so101 import ik_step

MODEL = os.path.join(MENAGERIE, "robotstudio_so101", "scene_box.xml")
STEP_M = 0.005          # metres of target motion per key press
STEP_GRIP = 0.10        # radians of jaw motion per key press
GRIP_RANGE = (-0.10, 0.90)
# scene_box.xml parks its box at x = 0.5 m. The SO-101's tool tip cannot get
# past x = 0.480 m at any joint configuration, so move it into reach.
BOX_START = np.array([0.36, 0.0, 0.03])

JOG = {
    "I": np.array([+STEP_M, 0.0, 0.0]), "K": np.array([-STEP_M, 0.0, 0.0]),
    "J": np.array([0.0, +STEP_M, 0.0]), "L": np.array([0.0, -STEP_M, 0.0]),
    "U": np.array([0.0, 0.0, +STEP_M]), "O": np.array([0.0, 0.0, -STEP_M]),
}
GRIP = {"G": -STEP_GRIP, "H": +STEP_GRIP}


class Teleop:
    """Every piece of mutable state in a teleoperation session, in one place."""

    def __init__(self, model, data):
        self.model, self.data = model, data
        self.offsets, self.site_offset = chain_geometry(model)
        self.lo = model.jnt_range[:5, 0]
        self.hi = model.jnt_range[:5, 1]
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self.q_cmd = np.zeros(5)
        self.grip = 0.3
        self.home = fk(self.q_cmd, self.offsets, self.site_offset)[:3, 3].copy()
        self.target = self.home.copy()

    def on_key(self, keycode):
        """The viewer hands us a GLFW key code; letters arrive as capital ASCII."""
        key = chr(keycode) if 0 < keycode < 128 else ""
        if key in JOG:
            self.target = self.target + JOG[key]
        elif key in GRIP:
            self.grip = float(np.clip(self.grip + GRIP[key], *GRIP_RANGE))
        elif key == "R":
            self.target = self.home.copy()

    def control(self):
        """One IK update, warm-started from the last command, then write ctrl."""
        self.q_cmd, _ = ik_step(self.target, self.q_cmd, self.offsets,
                                self.site_offset, self.lo, self.hi)
        self.data.ctrl[:5] = self.q_cmd
        self.data.ctrl[5] = self.grip

    def run(self, steps):
        for _ in range(steps):
            self.control()
            mujoco.mj_step(self.model, self.data)

    @property
    def tip(self):
        return np.array(self.data.site_xpos[self.site])

    @property
    def lag_mm(self):
        return float(np.linalg.norm(self.target - self.tip) * 1000)


def load():
    if not os.path.exists(MODEL):
        sys.exit(f"model not found: {MODEL}\n"
                 "clone MuJoCo Menagerie, or set MENAGERIE=/path/to/mujoco_menagerie")
    model = mujoco.MjModel.from_xml_path(MODEL)
    data = mujoco.MjData(model)
    data.qpos[6:9] = BOX_START          # the box's free joint: position ...
    data.qpos[9:13] = [1.0, 0.0, 0.0, 0.0]   # ... then orientation, as a quaternion
    mujoco.mj_forward(model, data)
    return model, data


def draw_target(viewer, target):
    """Put one small sphere in the viewer's user scene, at the target point."""
    viewer.user_scn.ngeom = 1
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[0],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([0.008, 0.0, 0.0]),
        pos=np.asarray(target, dtype=float),
        mat=np.eye(3).flatten(),
        rgba=np.array([0.92, 0.55, 0.13, 0.9]),
    )


def interactive():
    import mujoco.viewer

    model, data = load()
    teleop = Teleop(model, data)
    print(__doc__.split("Run:")[0])
    with mujoco.viewer.launch_passive(model, data, key_callback=teleop.on_key) as viewer:
        while viewer.is_running():
            tick = time.time()
            teleop.control()
            mujoco.mj_step(model, data)
            with viewer.lock():
                draw_target(viewer, teleop.target)
            viewer.sync()
            slack = model.opt.timestep - (time.time() - tick)
            if slack > 0:
                time.sleep(slack)


# A scripted keyboard session: (label, key, presses). One press is one tap of
# that key. Presses land 50 ms apart, which is a brisk but human key-repeat rate.
SCRIPT = [
    ("back off", "K", 20),
    ("descend", "O", 39),
    ("push", "I", 22),
]
PRESS_GAP = 10          # physics steps between presses: 50 ms at 200 Hz
RELEASE = 120           # steps to sit still after the last press: 0.6 s


def replay():
    model, data = load()
    teleop = Teleop(model, data)
    box = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    teleop.run(200)

    print(f"{'phase':<10}{'target (m)':<24}{'tool tip (m)':<24}"
          f"{'held':>7}{'released':>10}{'box x':>8}")
    print(f"{'':<10}{'':<24}{'':<24}{'mm':>7}{'mm':>10}{'mm':>8}")
    for label, key, presses in SCRIPT:
        for _ in range(presses):
            teleop.on_key(ord(key))
            teleop.run(PRESS_GAP)
        held = teleop.lag_mm
        teleop.run(RELEASE)
        print(f"{label:<10}{np.round(teleop.target, 3)!s:<24}"
              f"{np.round(teleop.tip, 3)!s:<24}{held:7.1f}{teleop.lag_mm:10.2f}"
              f"{data.xpos[box][0] * 1000:8.1f}")

    teleop.run(400)
    moved = (data.xpos[box][0] - BOX_START[0]) * 1000
    print(f"\nbox started at x = {BOX_START[0] * 1000:.0f} mm and ended at "
          f"{data.xpos[box][0] * 1000:.1f} mm: you pushed it {moved:.1f} mm.")


if __name__ == "__main__":
    replay() if "--replay" in sys.argv else interactive()
