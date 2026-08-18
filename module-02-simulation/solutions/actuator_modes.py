"""Solution - Lesson 2.7: MuJoCo actuator types on a one-joint bench.

Self-contained. Run:  python actuator_modes.py [a|b|c|all]
"""
import sys

import mujoco
import numpy as np

BENCH = """
<mujoco model="actuator-bench">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81" timestep="0.001" integrator="implicitfast"/>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <body name="link" pos="0 0 0.6">
      <joint name="elbow" type="hinge" axis="0 -1 0" damping="0.05"/>
      <geom name="rod" type="capsule" fromto="0 0 0 0 0 -0.4" size="0.008" mass="0"
            contype="0" conaffinity="0" rgba="0.55 0.55 0.6 1"/>
      <geom name="bob" type="sphere" pos="0 0 -0.4" size="0.03" mass="0.5"/>
    </body>
    {wall}
  </worldbody>
  <actuator>
    {act}
  </actuator>
</mujoco>
"""
WALL = '<geom name="wall" type="box" pos="0.24 0 0.28" size="0.02 0.06 0.12" rgba="0.62 0.34 0.26 1"/>'

MOTOR = '<motor    name="elbow_act" joint="elbow" gear="1" ctrlrange="-3 3"/>'
POSITION = '<position name="elbow_act" joint="elbow" kp="8" kv="1.2" ctrlrange="-3.2 3.2" forcerange="-3 3"/>'
VELOCITY = '<velocity name="elbow_act" joint="elbow" kv="4" ctrlrange="-3 3" forcerange="-3 3"/>'

KP, KV, FMAX = 8.0, 1.2, 3.0
M_G_L = 0.5 * 9.81 * 0.4        # the bob's gravity torque at horizontal, N-m


def build(actuator_xml, wall=False):
    return mujoco.MjModel.from_xml_string(
        BENCH.format(act=actuator_xml, wall=WALL if wall else ""))


def run(model, ctrl, seconds):
    """Hold `ctrl` constant for `seconds` of simulated time; return the MjData."""
    data = mujoco.MjData(model)
    data.ctrl[0] = ctrl
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    return data


def pd_force(target, q, qd):
    """MuJoCo's position-actuator law, reimplemented."""
    return float(np.clip(KP * (target - q) - KV * qd, -FMAX, FMAX))


def part_a():
    print("A. One number, three contracts (ctrl = 1.0 for 20 s)")
    for label, act in (("motor", MOTOR), ("position", POSITION), ("velocity", VELOCITY)):
        d = run(build(act), 1.0, 20.0)
        print(f"   {label:9s} angle {np.degrees(d.qpos[0]):9.3f} deg   "
              f"speed {np.degrees(d.qvel[0]):9.3f} deg/s   force {d.actuator_force[0]:+.4f} N-m")
    print("   ctrl means torque, target angle, target speed. The number never said which.\n")


def part_b():
    print("B. The position actuator is a PD controller you did not write")
    model = build(POSITION)
    data = mujoco.MjData(model)
    data.ctrl[0] = np.pi / 2                       # hold horizontal, as in lesson 1.17
    worst = 0.0
    for _ in range(6000):
        q, qd = data.qpos[0], data.qvel[0]         # state the step will act on
        mujoco.mj_step(model, data)
        worst = max(worst, abs(pd_force(data.ctrl[0], q, qd) - data.actuator_force[0]))
    settled = np.degrees(data.qpos[0])
    print(f"   biasprm = {model.actuator_biasprm[0][:3]}   (0, -kp, -kv)")
    print(f"   max |my PD - data.actuator_force| = {worst:.2e} N-m")
    print(f"   settles at {settled:.4f} deg, commanded 90     droop {90 - settled:.4f} deg")
    print(f"   holding torque {data.actuator_force[0]:.4f} = gravity torque "
          f"{M_G_L * np.sin(data.qpos[0]):.4f} N-m\n")
    assert worst < 1e-9, "the position actuator is exactly this PD law"


def part_c():
    print("C. Free space and a wall, same command (ctrl = 90 deg)")
    for label, wall in (("free   ", False), ("blocked", True)):
        model = build(POSITION, wall=wall)
        data = mujoco.MjData(model)
        data.ctrl[0] = np.pi / 2
        for k in range(6000):
            mujoco.mj_step(model, data)
            if k in (99, 499, 1999, 5999):
                print(f"   {label} t={data.time:4.1f}s  ctrl={np.degrees(data.ctrl[0]):6.2f}"
                      f"  qpos={np.degrees(data.qpos[0]):6.2f}"
                      f"  actuator_force={data.actuator_force[0]:+.3f}"
                      f"  contacts={data.ncon}")
    print("   ctrl is identical in both. qpos differs. Only actuator_force says why.\n")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("a", "all"):
        part_a()
    if which in ("b", "all"):
        part_b()
    if which in ("c", "all"):
        part_c()


if __name__ == "__main__":
    main()
