"""Lesson 2.5 - read the SO-101 Menagerie model from the outside.

Everything printed here is read back out of the compiled model, so it cannot
drift from the XML the way a hand-written table would.

Run:  python inspect_arm.py                 (the whole report)
      python inspect_arm.py --tree          (body tree only)
      python inspect_arm.py --limits        (actuator vs joint range audit)
      python inspect_arm.py --dof           (what each joint actually moves)
      python inspect_arm.py --menagerie ~/somewhere/else/mujoco_menagerie
"""
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

DEG = 180.0 / np.pi
MODEL = "robotstudio_so101/scene.xml"


def find_scene(argv, rel=MODEL):
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


def name_of(model, objtype, i):
    return mujoco.mj_id2name(model, objtype, i) or f"<unnamed {objtype.name} {i}>"


def body_tree(model):
    """Print the body hierarchy, which is exactly the XML's nesting."""
    print("body tree  (indent = XML nesting = kinematic chain)")
    children = {i: [] for i in range(model.nbody)}
    for i in range(1, model.nbody):
        children[model.body_parentid[i]].append(i)

    def walk(i, depth):
        njnt = model.body_jntnum[i]
        joints = [
            name_of(model, mujoco.mjtObj.mjOBJ_JOINT, model.body_jntadr[i] + k)
            for k in range(njnt)
        ]
        tag = f"joint: {', '.join(joints)}" if joints else "welded to parent"
        label = "  " * depth + name_of(model, mujoco.mjtObj.mjOBJ_BODY, i)
        print(f"  {label:<34}{model.body_mass[i] * 1000:6.1f} g   {tag}")
        for c in children[i]:
            walk(c, depth + 1)

    for c in children[0]:
        walk(c, 0)
    print(f"  total mass: {model.body_mass[1:].sum() * 1000:.0f} g")


JOINT_KIND = {
    mujoco.mjtJoint.mjJNT_FREE: "free",
    mujoco.mjtJoint.mjJNT_BALL: "ball",
    mujoco.mjtJoint.mjJNT_SLIDE: "slide (prismatic)",
    mujoco.mjtJoint.mjJNT_HINGE: "hinge (revolute)",
}


def joint_table(model):
    print("\njoints")
    print(f"  {'name':<16}{'kind':<20}{'range (deg)':<20}{'qpos':>5}{'qvel':>6}")
    for j in range(model.njnt):
        lo, hi = model.jnt_range[j]
        kind = JOINT_KIND[mujoco.mjtJoint(model.jnt_type[j])]
        span = f"{lo * DEG:7.1f} .. {hi * DEG:6.1f}" if model.jnt_limited[j] else "unlimited"
        print(f"  {name_of(model, mujoco.mjtObj.mjOBJ_JOINT, j):<16}{kind:<20}{span:<20}"
              f"{model.jnt_qposadr[j]:>5}{model.jnt_dofadr[j]:>6}")
    print(f"  nq = {model.nq} position numbers, nv = {model.nv} velocity numbers")


def actuator_table(model):
    print("\nactuators  (position servos: force = kp*(ctrl - qpos) - kv*qvel)")
    print(f"  {'name':<16}{'drives joint':<16}{'ctrlrange (deg)':<20}{'kp':>8}{'kv':>7}{'max N·m':>9}")
    for a in range(model.nu):
        jid = model.actuator_trnid[a, 0]
        lo, hi = model.actuator_ctrlrange[a]
        kp = model.actuator_gainprm[a, 0]
        kv = -model.actuator_biasprm[a, 2]
        fmax = model.actuator_forcerange[a, 1]
        span = f"{lo * DEG:7.1f} .. {hi * DEG:6.1f}"
        print(f"  {name_of(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a):<16}"
              f"{name_of(model, mujoco.mjtObj.mjOBJ_JOINT, jid):<16}{span:<20}"
              f"{kp:8.2f}{kv:7.3f}{fmax:9.2f}")


def limit_audit(model):
    """Compare every actuator's ctrlrange against the range of the joint it drives."""
    print("\nctrlrange vs joint range")
    bad = 0
    for a in range(model.nu):
        jid = model.actuator_trnid[a, 0]
        clo, chi = model.actuator_ctrlrange[a]
        jlo, jhi = model.jnt_range[jid]
        over = max(jlo - clo, chi - jhi, 0.0)
        flag = ""
        if over > 1e-6:
            bad += 1
            flag = f"  <-- commandable past the joint stop by {over * DEG:.2f} deg"
        print(f"  {name_of(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a):<16}"
              f"ctrl [{clo * DEG:7.2f},{chi * DEG:7.2f}]  joint [{jlo * DEG:7.2f},{jhi * DEG:7.2f}]{flag}")
    print(f"  {bad} actuator(s) can be asked for an angle the joint will not reach.")
    return bad


def geom_report(model):
    """Every link carries two skins: one you look at, one that collides."""
    print("\ngeoms  (group 2 = visual meshes, groups 3-4 = collision shapes)")
    groups = {}
    for g in range(model.ngeom):
        groups.setdefault(int(model.geom_group[g]), []).append(g)
    for grp in sorted(groups):
        ids = groups[grp]
        collides = sum(
            1 for g in ids if model.geom_contype[g] or model.geom_conaffinity[g]
        )
        print(f"  group {grp}: {len(ids):3d} geoms, {collides:3d} of them collide")
    total = sum(
        1 for g in range(model.ngeom)
        if model.geom_contype[g] or model.geom_conaffinity[g]
    )
    print(f"  {total} of {model.ngeom} geoms take part in contact.")


def dof_report(model):
    """Nudge each joint and measure how far the fingertip site travels."""
    data = mujoco.MjData(model)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    mujoco.mj_forward(model, data)
    home = data.site_xpos[site].copy()

    print("\nwhat each joint moves  (fingertip travel for a 0.30 rad nudge)")
    positioning = 0
    for j in range(model.njnt):
        data.qpos[:] = 0
        data.qpos[model.jnt_qposadr[j]] = 0.30
        mujoco.mj_forward(model, data)
        moved = float(np.linalg.norm(data.site_xpos[site] - home))
        if moved > 1e-6:
            positioning += 1
        print(f"  {name_of(model, mujoco.mjtObj.mjOBJ_JOINT, j):<16}{moved * 1000:7.1f} mm")
    jaw = model.njnt - positioning
    print(f"  {positioning} joints move the hand; the other {jaw} only works the jaw.")

    # A full pose is 6 numbers. Stack the fingertip's position and orientation
    # Jacobian and ask how many of those 6 the arm can independently touch.
    rng = np.random.default_rng(0)
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    ranks = []
    for _ in range(200):
        data.qpos[:] = rng.uniform(model.jnt_range[:, 0], model.jnt_range[:, 1])
        mujoco.mj_forward(model, data)
        mujoco.mj_jacSite(model, data, jacp, jacr, site)
        ranks.append(int(np.linalg.matrix_rank(np.vstack([jacp, jacr])[:, :positioning], tol=1e-6)))
    print(f"  fingertip Jacobian rank over 200 random poses: {sorted(set(ranks))} out of 6")
    print("  rank < 6 everywhere means some pose is unreachable from every configuration.")


def main(argv):
    path = find_scene(argv)
    model = mujoco.MjModel.from_xml_path(str(path))
    print(f"{path}\n{'-' * 72}")
    all_ = not any(f in argv for f in ("--tree", "--limits", "--dof"))
    if all_ or "--tree" in argv:
        body_tree(model)
    if all_:
        joint_table(model)
        actuator_table(model)
        geom_report(model)
    if all_ or "--limits" in argv:
        limit_audit(model)
    if all_ or "--dof" in argv:
        dof_report(model)


if __name__ == "__main__":
    main(sys.argv[1:])
