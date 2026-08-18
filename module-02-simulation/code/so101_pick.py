"""Shared rig for Lessons 2.11-2.13: load the task scene, drive the SO-101, pick and place.

This file is given to you complete. It is the machinery the three exercises sit on
top of, not an exercise itself.

Point MUJOCO_MENAGERIE at your clone if it is not in ~/mujoco_menagerie.

Run:  python so101_pick.py            one scripted episode, prints the verdict
"""
import os
import shutil
from pathlib import Path

import mujoco
import numpy as np

MENAGERIE = Path(os.environ.get("MUJOCO_MENAGERIE", Path.home() / "mujoco_menagerie"))
MODEL_DIR = MENAGERIE / "robotstudio_so101"
SCENE_SRC = Path(__file__).with_name("task_scene.xml")

OPEN, CLOSE = 1.2, -0.1          # gripper actuator commands, radians
GRASP_BACK = 0.008               # move the target this far back along the reach
                                 # direction: the site sits at the jaw tip, and the
                                 # fixed finger needs clearance on the way down
BIN_HALF = 0.05                  # success box, slightly inside the tray walls
CARRY_Z = 0.075


def install_scene(src=SCENE_SRC):
    """Copy the scene next to so101.xml and return the installed path.

    MJCF resolves <include> and mesh paths against the directory of the file
    that names them, so a scene living anywhere else cannot find the meshes.
    Menagerie's own scene.xml sits in the model directory for the same reason.
    """
    if not MODEL_DIR.is_dir():
        raise SystemExit(
            f"Menagerie model not found at {MODEL_DIR}.\n"
            "git clone https://github.com/google-deepmind/mujoco_menagerie ~/mujoco_menagerie\n"
            "or set MUJOCO_MENAGERIE to your clone."
        )
    dst = MODEL_DIR / src.name
    shutil.copyfile(src, dst)
    return dst


def target_frame(x, y):
    """Gripper pointing straight down, jaws opening along the reach direction."""
    phi = np.arctan2(y, x)
    return np.column_stack([[0.0, 0.0, -1.0],
                            [-np.sin(phi), np.cos(phi), 0.0],
                            [np.cos(phi), np.sin(phi), 0.0]])


def ik(model, scratch, site_id, pos, q0, iters=40, lam=0.08, alpha=0.6):
    """Damped least squares on the gripper site, 5 arm joints. Same method as Lesson 1.15."""
    R = target_frame(pos[0], pos[1])
    q = np.array(q0, dtype=float)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    for _ in range(iters):
        scratch.qpos[:5] = q
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)
        err_p = pos - scratch.site_xpos[site_id]
        R_err = R @ scratch.site_xmat[site_id].reshape(3, 3).T
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, R_err.flatten())
        err_r = np.zeros(3)
        mujoco.mju_quat2Vel(err_r, quat, 1.0)
        if np.linalg.norm(err_p) < 1e-4 and np.linalg.norm(err_r) < 1e-3:
            break
        mujoco.mj_jacSite(model, scratch, jacp, jacr, site_id)
        J = np.vstack([jacp[:, :5], jacr[:, :5]])
        err = np.concatenate([err_p, err_r])
        q = q + alpha * (J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), err))
        q = np.clip(q, model.jnt_range[:5, 0], model.jnt_range[:5, 1])
    return q, np.linalg.norm(err_p), np.linalg.norm(err_r)


class Arm:
    """The scene plus a tiny Cartesian move primitive."""

    def __init__(self, path=None):
        self.model = mujoco.MjModel.from_xml_path(str(path or install_scene()))
        self.data = mujoco.MjData(self.model)
        self._scratch = mujoco.MjData(self.model)
        self.site = self.model.site("gripperframe").id
        self.reset()

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self.q = self.data.ctrl[:5].copy()
        self.pos = self.data.site_xpos[self.site].copy()
        self.trace = None            # set to [] to record qpos every step

    def _step(self):
        mujoco.mj_step(self.model, self.data)
        if self.trace is not None:
            self.trace.append(self.data.qpos[:6].copy())

    def place_cube(self, x, y, z=0.02):
        self.data.qpos[6:9] = (x, y, z)
        self.data.qpos[9:13] = (1, 0, 0, 0)
        self.data.qvel[-6:] = 0
        mujoco.mj_forward(self.model, self.data)

    def cube(self):
        return self.data.body("cube").xpos.copy()

    def goto(self, pos, grip, seconds, knots=16):
        """Straight line in Cartesian space. IK at the knots, interpolate between them.

        Interpolating joint angles instead makes the path bow outward, and the
        fixed finger clips the cube on the way down.
        """
        pos = np.asarray(pos, dtype=float)
        qs = [self.q.copy()]
        for k in range(1, knots + 1):
            target = self.pos + (k / knots) * (pos - self.pos)
            qs.append(ik(self.model, self._scratch, self.site, target, qs[-1])[0])
        qs = np.array(qs)
        grip0 = self.data.ctrl[5]
        steps = int(seconds / self.model.opt.timestep)
        for i in range(steps):
            s = (i + 1) / steps
            s = 3 * s * s - 2 * s ** 3                     # ease in, ease out
            f = s * knots
            k0 = int(np.floor(f))
            k1 = min(k0 + 1, knots)
            self.data.ctrl[:5] = qs[k0] + (f - k0) * (qs[k1] - qs[k0])
            self.data.ctrl[5] = grip0 + s * (grip - grip0)
            self._step()
        self.q = qs[-1].copy()
        self.pos = pos.copy()

    def hold(self, seconds):
        for _ in range(int(seconds / self.model.opt.timestep)):
            self._step()

    def touching_gripper(self):
        """True if any contact involves the cube and a gripper geom."""
        cube_geom = self.model.geom("cube").id
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = (c.geom1, c.geom2)
            if cube_geom not in pair:
                continue
            other = pair[0] if pair[1] == cube_geom else pair[1]
            if self.model.geom_priority[other] == 1:       # the gripper's collision class
                return True
        return False


def cube_in_bin(arm, v_tol=0.02):
    """Three conditions, all of them necessary.

    Inside the tray footprint, low enough to be resting in it, and at rest and
    released. Drop any one and the detector scores something that is not a success.
    """
    # TODO(you): three conditions, all necessary.
    #   1. arm.cube()[:2] within BIN_HALF of arm.model.body("bin").pos[:2]
    #   2. arm.cube()[2] low enough to be resting in the tray, not held over it
    #   3. the cube is at rest (norm of arm.data.qvel[-6:] under v_tol)
    #      and not touching the gripper (arm.touching_gripper())
    raise NotImplementedError


def radial_back(p, distance):
    phi = np.arctan2(p[1], p[0])
    return np.array([p[0] - distance * np.cos(phi), p[1] - distance * np.sin(phi), p[2]])


def pick_and_place(arm, carry_seconds=1.2, carry_z=CARRY_Z, grasp_back=GRASP_BACK):
    """Scripted expert. It reads the true cube pose, which no real robot can do."""
    arm.hold(0.3)                                          # let the cube settle
    c = arm.cube()
    pre = radial_back(np.array([c[0], c[1], c[2] + 0.07]), grasp_back)
    grab = radial_back(np.array([c[0], c[1], c[2]]), grasp_back)
    bin_xy = arm.model.body("bin").pos[:2]      # one source of truth: the scene
    over = np.array([bin_xy[0], bin_xy[1], carry_z])
    arm.goto(pre, OPEN, 1.0)
    arm.goto(grab, OPEN, 0.8)
    arm.goto(grab, CLOSE, 0.6)
    arm.goto(pre, CLOSE, 0.7)
    lifted = arm.cube()[2] > 0.05
    arm.goto(over, CLOSE, carry_seconds)
    arm.goto(over, OPEN, 0.4)
    arm.goto(np.array([bin_xy[0] * 0.6, bin_xy[1] * 0.6, 0.11]), OPEN, 0.6)
    arm.hold(1.0)
    return dict(lifted=bool(lifted), success=cube_in_bin(arm), cube=arm.cube())


if __name__ == "__main__":
    arm = Arm()
    out = pick_and_place(arm)
    print(f"lifted={out['lifted']}  success={out['success']}  cube={np.round(out['cube'], 4)}")
