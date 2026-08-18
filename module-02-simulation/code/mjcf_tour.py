"""Lesson 2.4 - read an MJCF file the way the compiler read it.

Finish two_link.xml first (four TODOs in there), then the two TODOs here.

Run:  python mjcf_tour.py                 (tours ./two_link.xml)
      python mjcf_tour.py some/other.xml
"""
import sys

import mujoco
import numpy as np

JOINT_KIND = {
    mujoco.mjtJoint.mjJNT_FREE: "free   (6 DOF, nq=7)",
    mujoco.mjtJoint.mjJNT_BALL: "ball   (3 DOF, nq=4)",
    mujoco.mjtJoint.mjJNT_SLIDE: "slide  (1 DOF, nq=1)",
    mujoco.mjtJoint.mjJNT_HINGE: "hinge  (1 DOF, nq=1)",
}


def children_of(model):
    """Map parent body id -> list of child body ids. Body 0 is the world and parents itself."""
    kids = {i: [] for i in range(model.nbody)}
    for i in range(1, model.nbody):
        kids[int(model.body(i).parentid[0])].append(i)
    return kids


def print_tree(model, body=0, depth=0, kids=None):
    """Print <worldbody> as the tree it is: indentation is parenthood.

    For each body print its name, its pos (which is relative to its PARENT, not the
    world) and the mass the compiler worked out for you. Then, indented under it,
    every joint whose bodyid is this body, every geom, every site. Then recurse into
    children_of(model)[body].

    Useful accessors, all of which take an integer index:
        model.body(i).name / .pos / .mass / .parentid
        model.joint(j).name / .type / .axis / .bodyid / .qposadr
        model.geom(g).name / .type / .bodyid / .condim
        model.site(s).name / .bodyid
    Enum to string:  mujoco.mjtGeom(int(gm.type[0])).name  ->  'mjGEOM_CAPSULE'
    """
    # TODO(you)
    raise NotImplementedError


def print_flat_lists(model):
    """Actuators, sensors and keyframes live outside the tree, in flat ordered lists."""
    print("\nactuator  (index == the slot you write in data.ctrl)")
    for i in range(model.nu):
        a = model.actuator(i)
        joint = model.joint(int(a.trnid[0])).name
        print(f"  ctrl[{i}] {a.name!r} -> joint {joint!r}  kp={a.gainprm[0]:.1f}"
              f"  ctrlrange={np.round(a.ctrlrange, 2)}")

    print("\nsensor    (index into data.sensordata)")
    for i in range(model.nsensor):
        s = model.sensor(i)
        print(f"  sensordata[{int(s.adr[0])}:{int(s.adr[0]) + int(s.dim[0])}] {s.name!r}"
              f"  {mujoco.mjtSensor(int(s.type[0])).name.replace('mjSENS_', '').lower()}")

    print("\nkeyframe")
    for i in range(model.nkey):
        print(f"  {model.key(i).name!r}  qpos={np.round(model.key(i).qpos, 3)}")


def check(model):
    """Fail loudly and specifically if a TODO in the XML is still open."""
    def need(cond, msg):
        if not cond:
            raise SystemExit(f"two_link.xml is incomplete: {msg}")

    names = [model.body(i).name for i in range(model.nbody)]
    need("forearm" in names, "no body named 'forearm' (TODO 1)")
    need(model.body("forearm").parentid[0] == model.body("upper_arm").id,
         "'forearm' exists but its parent is not 'upper_arm' - it is nested in the wrong place (TODO 1)")
    need(model.nu == 2, f"expected 2 actuators, found {model.nu} (TODO 2)")
    need(model.nsensor == 2, f"expected 2 sensors, found {model.nsensor} (TODO 3)")
    need(model.nkey == 1, f"expected 1 keyframe, found {model.nkey} (TODO 4)")
    print("\nstructure check passed.")


def run(model, seconds=3.0):
    """Reset to the keyframe, command a pose, step, and report what actually happened.

    Steps:
      data = mujoco.MjData(model)
      mujoco.mj_resetDataKeyframe(model, data, 0)   # jump to <key name="home">
      mujoco.mj_forward(model, data)                # positions without integrating
      print the tip site's world position: data.site("tip").xpos
      write data.ctrl[:] = [-0.6, 1.2]              # two position actuators, radians
      step with mujoco.mj_step(model, data) until data.time >= seconds
      print commanded vs data.qpos[:2], and the difference
      print data.site("tip").xpos and data.sensor("tip_pos").data - they should agree
    """
    # TODO(you)
    raise NotImplementedError


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "two_link.xml"
    model = mujoco.MjModel.from_xml_path(path)
    print(f"{path}: {model.nbody} bodies, {model.njnt} joints, {model.ngeom} geoms, "
          f"nq={model.nq}, nv={model.nv}, nu={model.nu}\n")
    print_tree(model)
    print_flat_lists(model)
    check(model)
    run(model)
