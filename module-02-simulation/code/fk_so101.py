"""Lesson 2.9 - your own forward kinematics for the SO-101, checked against
MuJoCo's.

Everything you need is Module 1: quaternion to rotation matrix (Lesson 1.7),
homogeneous transforms (1.8), chain composition (1.9). MuJoCo supplies only the
link geometry, which is what a model file is for.

Run:  python fk_so101.py
"""
import os
import sys

import numpy as np
import mujoco

MENAGERIE = os.environ.get("MENAGERIE", os.path.expanduser("~/mujoco_menagerie"))
MODEL = os.path.join(MENAGERIE, "robotstudio_so101", "scene.xml")

# The kinematic chain of the SO-101, base to gripper. Five moving joints; the
# sixth actuator ("gripper") drives a jaw hanging off the gripper body and does
# not move the wrist, so it is not part of this chain.
CHAIN = ["base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper"]
EE_SITE = "gripperframe"


def load():
    if not os.path.exists(MODEL):
        sys.exit(f"model not found: {MODEL}\n"
                 "clone MuJoCo Menagerie, or set MENAGERIE=/path/to/mujoco_menagerie")
    model = mujoco.MjModel.from_xml_path(MODEL)
    return model, mujoco.MjData(model)


def quat_to_R(q):
    """MuJoCo quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    w, x, y, z = q
    # TODO(you): the standard quaternion-to-matrix formula from Lesson 1.7.
    #            Row 0 is [1 - 2(y² + z²), 2(xy - zw), 2(xz + yw)].
    raise NotImplementedError


def T(R, p):
    """4x4 homogeneous transform from a rotation and a translation."""
    out = np.eye(4)
    out[:3, :3] = R
    out[:3, 3] = p
    return out


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def chain_geometry(model):
    """Pull the fixed link offsets out of the model, once.

    Returns (offsets, site_offset). offsets[k] is the parent-to-child transform
    of CHAIN[k] with its joint at zero; site_offset is gripper-body to tool tip.
    """
    offsets = []
    for name in CHAIN:
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        offsets.append(T(quat_to_R(model.body_quat[b]), model.body_pos[b]))
    s = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    return offsets, T(quat_to_R(model.site_quat[s]), model.site_pos[s])


def fk(q, offsets, site_offset):
    """Pose of the tool tip in world coordinates, for five joint angles.

    Every SO-101 joint is a hinge about its own body's z axis with the anchor at
    the body origin, so each joint contributes exactly one Rz.
    """
    pose = np.eye(4)
    # TODO(you): walk the chain. For each offset in order:
    #              pose = pose @ offset
    #            and then, for every body except 'base' (which is welded to the
    #            world and has no joint):
    #              pose = pose @ T(Rz(q[k - 1]), np.zeros(3))
    #            Finish with the site offset and return the 4x4.
    raise NotImplementedError


def mujoco_fk(model, data, q, site_id):
    """MuJoCo's answer, for comparison."""
    data.qpos[:5] = q
    data.qpos[5] = 0.0
    mujoco.mj_forward(model, data)
    return np.array(data.site_xpos[site_id]), np.array(data.site_xmat[site_id]).reshape(3, 3)


def main(n=2000):
    model, data = load()
    offsets, site_offset = chain_geometry(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    lo, hi = model.jnt_range[:5, 0], model.jnt_range[:5, 1]

    home = np.zeros(5)
    mine = fk(home, offsets, site_offset)
    theirs_p, theirs_R = mujoco_fk(model, data, home, site_id)
    print("all joints at zero")
    print(f"  yours   {np.round(mine[:3, 3], 6)}")
    print(f"  MuJoCo  {np.round(theirs_p, 6)}")

    rng = np.random.default_rng(0)
    worst_p = worst_R = 0.0
    for _ in range(n):
        q = rng.uniform(lo, hi)
        mine = fk(q, offsets, site_offset)
        p, R = mujoco_fk(model, data, q, site_id)
        worst_p = max(worst_p, float(np.linalg.norm(mine[:3, 3] - p)))
        worst_R = max(worst_R, float(np.abs(mine[:3, :3] - R).max()))
    print(f"\n{n} random poses inside the joint limits")
    print(f"  worst position disagreement  {worst_p:.3e} m")
    print(f"  worst rotation disagreement  {worst_R:.3e} (matrix element)")
    assert worst_p < 1e-12 and worst_R < 1e-12
    print("\nOK: same kinematics, two independent implementations.")


if __name__ == "__main__":
    main()
