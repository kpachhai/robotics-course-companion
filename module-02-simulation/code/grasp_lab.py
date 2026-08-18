"""Lesson 2.12 - look at what the contact solver is actually doing, then break the grasp.

Run:  python grasp_lab.py            all five experiments  (about 4 minutes)
      python grasp_lab.py --quick    the matrix at n=5 instead of n=20
"""
import sys

import mujoco
import numpy as np

import so101_pick as sp

PYRAMIDAL = mujoco.mjtCone.mjCONE_PYRAMIDAL
ELLIPTIC = mujoco.mjtCone.mjCONE_ELLIPTIC


def penetration():
    """Rest the cube on the table and measure how far into it the cube sits."""
    print("penetration at rest (cube half-height 0.020 m, so z < 0.020 means overlap)")
    for timeconst, mass in [(0.01, 0.064), (0.01, 0.64), (0.02, 0.064), (0.002, 0.064)]:
        arm = sp.Arm()
        arm.model.geom_solref[arm.model.geom("cube").id] = (timeconst, 1.0)
        arm.model.body_mass[arm.model.body("cube").id] = mass
        arm.place_cube(0.22, 0.0, 0.025)
        arm.hold(2.0)
        z = arm.cube()[2]
        dists = [arm.data.contact[i].dist for i in range(arm.data.ncon)]
        print(f"  solref timeconst {timeconst:5.3f} s, mass {mass:5.3f} kg -> "
              f"cube z {z * 1000:6.3f} mm, sinks {20 - z * 1000:5.3f} mm, "
              f"deepest contact {min(dists) * 1000:6.3f} mm")


def contact_report():
    """List every contact on the cube at the moment the grasp closes."""
    arm = sp.Arm()
    c = arm.cube()
    arm.goto(sp.radial_back(np.array([c[0], c[1], c[2] + 0.07]), sp.GRASP_BACK), sp.OPEN, 1.0)
    arm.goto(sp.radial_back(np.array([c[0], c[1], c[2]]), sp.GRASP_BACK), sp.OPEN, 0.8)
    arm.goto(arm.pos, sp.CLOSE, 0.6)
    cube_geom = arm.model.geom("cube").id
    print(f"\ncontacts while gripping: {arm.data.ncon} total in the scene")
    forces = np.zeros(6)
    for i in range(arm.data.ncon):
        con = arm.data.contact[i]
        if cube_geom not in (con.geom1, con.geom2):
            continue
        other = con.geom1 if con.geom2 == cube_geom else con.geom2
        name = mujoco.mj_id2name(arm.model, mujoco.mjtObj.mjOBJ_GEOM, other) or f"<unnamed geom {other}>"
        mujoco.mj_contactForce(arm.model, arm.data, i, forces)
        print(f"  cube <-> {name:22s}"
              f" dist {con.dist * 1000:7.3f} mm  dim {con.dim}"
              f"  mu {con.friction[0]:.3f}  normal force {forces[0]:6.3f} N")


def priority_demo():
    """The friction you wrote on a geom is not necessarily the friction the solver uses."""
    print("\nwhose friction wins? (cube mu, finger mu, gripper priority)")
    trials = [("1.00  1.00  on   (as shipped)", 1.00, 1.00, False),
              ("0.05  1.00  on ", 0.05, 1.00, False),
              ("0.05  1.00  off", 0.05, 1.00, True),
              ("1.00  0.05  on ", 1.00, 0.05, False),
              ("1.00  0.05  off", 1.00, 0.05, True)]
    for label, cube_mu, finger_mu, drop_priority in trials:
        arm = sp.Arm()
        arm.model.geom_friction[arm.model.geom("cube").id, 0] = cube_mu
        arm.model.geom_friction[arm.model.geom_priority == 1, 0] = finger_mu
        if drop_priority:
            arm.model.geom_priority[:] = 0
        c = arm.cube()
        arm.goto(sp.radial_back(np.array([c[0], c[1], c[2] + 0.07]), sp.GRASP_BACK), sp.OPEN, 1.0)
        arm.goto(sp.radial_back(np.array([c[0], c[1], c[2]]), sp.GRASP_BACK), sp.OPEN, 0.8)
        arm.goto(arm.pos, sp.CLOSE, 0.6)
        cube_geom = arm.model.geom("cube").id
        mus = set()
        for i in range(arm.data.ncon):
            con = arm.data.contact[i]
            if cube_geom not in (con.geom1, con.geom2):
                continue
            other = con.geom1 if con.geom2 == cube_geom else con.geom2
            if other != arm.model.geom("table_top").id:
                mus.add(round(float(con.friction[0]), 3))
        print(f"  {label:32s} finger contact mu used: {sorted(mus)}")


def grasp_window():
    """How far can the approach point be off before the grasp fails?

    Positive means the gripper stops short of the cube, negative means it
    reaches past it. Two parallel plates would give a symmetric window.
    """
    print("\ngrasp tolerance along the reach direction (nominal offset 8 mm)")
    line = []
    for back in np.arange(-0.006, 0.045, 0.003):
        arm = sp.Arm()
        ok = sp.pick_and_place(arm, grasp_back=float(back))["success"]
        line.append((round(float(back) * 1000, 1), ok))
    for offset, ok in line:
        print(f"  offset {offset:+6.1f} mm  {'place' if ok else 'FAIL '}")
    good = [o for o, ok in line if ok]
    print(f"  window: {min(good):+.1f} mm to {max(good):+.1f} mm, "
          f"{max(good) - min(good):.1f} mm wide, nominal sits {8 - min(good):.1f} mm from one edge "
          f"and {max(good) - 8:.1f} mm from the other")


def episode(cone, impratio, grip_mu, mass, carry_seconds, cube_xy):
    arm = sp.Arm()
    # TODO(you): set the friction cone and the friction-to-normal impedance
    # ratio on arm.model.opt, then set the finger friction. Every geom with
    # geom_priority == 1 belongs to the gripper's collision class.
    raise NotImplementedError
    arm.model.body_mass[arm.model.body("cube").id] = mass
    arm.place_cube(*cube_xy)
    out = sp.pick_and_place(arm, carry_seconds=carry_seconds)
    return out["lifted"], out["success"]


def contact_matrix(n=20, seed=1, grip_mu=0.12, mass=0.4, carry_seconds=0.5):
    """The same pick, on a heavy slippery object, under four contact settings."""
    print(f"\ncontact-mode matrix: cube {mass * 1000:.0f} g, finger mu {grip_mu}, "
          f"carry in {carry_seconds} s, n = {n} per row")
    configs = [("elliptic, impratio 10 (as shipped)", ELLIPTIC, 10.0),
               ("elliptic, impratio 1", ELLIPTIC, 1.0),
               ("pyramidal, impratio 10", PYRAMIDAL, 10.0),
               ("pyramidal, impratio 1 (MuJoCo defaults)", PYRAMIDAL, 1.0)]
    for label, cone, impratio in configs:
        rng = np.random.default_rng(seed)
        lifted = placed = 0
        for _ in range(n):
            xy = (0.22 + rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02), 0.02)
            got_up, ok = episode(cone, impratio, grip_mu, mass, carry_seconds, xy)
            lifted += got_up
            placed += ok
        print(f"  {label:42s} lifted {lifted:2d}/{n}   in bin {placed:2d}/{n}")


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    penetration()
    contact_report()
    priority_demo()
    grasp_window()
    contact_matrix(n=5 if quick else 20)
