"""Lesson 2.5 - read the SO-101 Menagerie model from the outside.

Fill in the three TODOs. Everything you print must be read back out of the
compiled model, never typed in by hand - that is the whole point: a hand-written
table goes stale the moment upstream edits the XML, and a printed one cannot.

Run:  python inspect_arm.py                 (the whole report)
      python inspect_arm.py --tree          (body tree only)
      python inspect_arm.py --limits        (actuator vs joint range audit)
      python inspect_arm.py --dof           (what each joint actually moves)
      python inspect_arm.py --menagerie ~/somewhere/else/mujoco_menagerie

Useful model fields:
  model.nbody / njnt / nu / ngeom / nq / nv
  model.body_parentid[i]   parent body index; body 0 is always the world
  model.body_jntnum[i]     how many joints attach this body to its parent
  model.body_jntadr[i]     index of this body's first joint
  model.jnt_type[j]        mujoco.mjtJoint member (hinge, slide, ball, free)
  model.jnt_range[j]       (low, high); meaningful when model.jnt_limited[j]
  model.jnt_qposadr[j]     where this joint's numbers start inside data.qpos
  model.actuator_trnid[a, 0]      the joint this actuator drives
  model.actuator_ctrlrange[a]     what you are allowed to write into data.ctrl
  mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
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
    """Print the body hierarchy, which is exactly the XML's nesting.

    One line per body, indented by depth: name, mass in grams, and either the
    joint that connects it to its parent or the words "welded to parent".
    Walk down from the world body (index 0) using model.body_parentid.
    """
    # TODO(you)
    raise NotImplementedError


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
    """Compare every actuator's ctrlrange against the range of the joint it drives.

    Print one line per actuator and flag any whose ctrlrange reaches past the
    joint's own range, in degrees. Return how many you flagged. Exactly one
    actuator on this arm fails; find it without looking it up.
    """
    # TODO(you)
    raise NotImplementedError


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
    """Nudge each joint and measure how far the fingertip site travels.

    Build an MjData, call mujoco.mj_forward, and record data.site_xpos for the
    site named "gripperframe". Then for each joint: zero qpos, set that one
    joint to 0.30 rad, mj_forward again, and print the distance the site moved.
    Count how many joints move it at all.

    Then the harder half. A full pose is six numbers, so ask how many of them
    this arm can independently reach for. mujoco.mj_jacSite(model, data, jacp,
    jacr, site_id) fills two 3-by-nv arrays: how the site's position and its
    orientation change per unit of joint velocity. Stack them into a 6-by-nv
    matrix, keep the columns for the positioning joints, and take its rank at a
    few hundred random configurations. If the rank is never 6, some pose is
    unreachable from every configuration - which is a claim about the metal,
    not about your code.
    """
    # TODO(you)
    raise NotImplementedError


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
