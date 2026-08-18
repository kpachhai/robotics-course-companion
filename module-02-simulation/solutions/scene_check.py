"""Lesson 2.11 - inspect the task scene, map what the arm can reach, test the success detector.

Run:  python scene_check.py
"""
import mujoco
import numpy as np

import so101_pick as sp


def inventory(arm):
    m = arm.model
    print("bodies    :", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)])
    print("actuators :", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)])
    print(f"nq={m.nq}  nv={m.nv}  nu={m.nu}   (6 arm joints + 1 free joint = {m.nq} qpos, {m.nv} qvel)")
    adr = m.body("cube").jntadr[0]
    print(f"cube free joint: qpos[{m.jnt_qposadr[adr]}:{m.jnt_qposadr[adr] + 7}] "
          f"= xyz + quaternion, qvel[{m.jnt_dofadr[adr]}:{m.jnt_dofadr[adr] + 6}] = linear + angular")


def reach_map(arm, n=25, z=0.02, pos_tol=2e-3, rot_tol=0.05):
    """Where on the table can the gripper stand vertically over a cube?

    Kinematics only. No self-collision, no table collision, so this is an
    optimistic upper bound on where the task is possible.
    """
    m = arm.model
    table = m.body("table").pos
    half = m.geom("table_top").size[0]
    xs = np.linspace(table[0] - half, table[0] + half, n)
    ys = np.linspace(table[1] - half, table[1] + half, n)
    ok = np.zeros((n, n), dtype=bool)
    scratch = mujoco.MjData(m)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            if np.hypot(x, y) < 0.05:
                continue
            _, ep, er = sp.ik(m, scratch, arm.site, np.array([x, y, z]),
                              arm.data.ctrl[:5], iters=120)
            ok[i, j] = ep < pos_tol and er < rot_tol
    cell = (xs[1] - xs[0]) * (ys[1] - ys[0])
    print(f"top-down grasp poses solvable on {ok.mean() * 100:.0f}% of the table "
          f"({ok.sum() * cell * 1e4:.0f} cm^2 of {ok.size * cell * 1e4:.0f} cm^2)")
    for i in range(n - 1, -1, -1):
        print("   " + "".join("#" if v else "." for v in ok[i]))
    return ok


def carry_ceiling(arm, radius=0.20, pos_tol=2e-3, rot_tol=0.05):
    """How high can the gripper go while still pointing straight down?"""
    scratch = mujoco.MjData(arm.model)
    best = 0.0
    for z in np.arange(0.02, 0.20, 0.005):
        _, ep, er = sp.ik(arm.model, scratch, arm.site, np.array([radius, 0.0, z]),
                          arm.data.ctrl[:5], iters=120)
        if ep < pos_tol and er < rot_tol:
            best = z
    print(f"highest vertical-gripper pose at radius {radius} m: z = {best:.3f} m "
          f"-> the bin walls must stay below that")
    return best


def detector_cases(arm):
    """Each case removes exactly one condition from the success predicate."""
    cases = [
        ("cube on the table where it started", (0.22, 0.0, 0.02), np.zeros(6), False),
        ("cube resting in the bin", (0.05, -0.19, 0.028), np.zeros(6), True),
        ("cube in the bin but still moving", (0.05, -0.19, 0.028), np.array([0, 0, -0.5, 0, 0, 0]), False),
        ("cube hovering above the bin", (0.05, -0.19, 0.09), np.zeros(6), False),
        ("cube just outside the bin wall", (0.05, -0.30, 0.02), np.zeros(6), False),
    ]
    for label, pos, vel, expected in cases:
        arm.reset()
        arm.place_cube(*pos)
        arm.data.qvel[-6:] = vel
        mujoco.mj_forward(arm.model, arm.data)
        got = sp.cube_in_bin(arm)
        flag = "ok " if got == expected else "BAD"
        print(f"  {flag} {label:38s} -> {got} (expected {expected})")
        assert got == expected, label


if __name__ == "__main__":
    arm = sp.Arm()
    inventory(arm)
    print()
    reach_map(arm)
    print()
    carry_ceiling(arm)
    print("\nsuccess detector:")
    detector_cases(arm)
    print("\none scripted episode:")
    arm.reset()
    out = sp.pick_and_place(arm)
    print(f"  lifted={out['lifted']}  success={out['success']}  cube={np.round(out['cube'], 4)}")
