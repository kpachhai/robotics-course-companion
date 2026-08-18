"""Lesson 2.8 - declaring sensors and rendering cameras.

The scene, the descent and the plots are done. Three things are yours: two
sensor declarations in the XML, the named read, and the depth render.

Run:  python sensors_and_cameras.py            # everything
      python sensors_and_cameras.py layout     # just the sensordata map
Writes sensor_trace.png and sensor_views.png next to itself.
"""
import statistics
import sys
import time
from pathlib import Path

import mujoco
import numpy as np


def beside(name, path=None):
    """Resolve an output file next to this script rather than in the cwd.

    Run from the repo root, a bare filename drops an untracked PNG there, and
    only `module-*/code/*.png` is gitignored.
    """
    return Path(__file__).with_name(name) if path is None else Path(path)


XML = """
<mujoco model="sensor-bench">
  <compiler angle="radian" autolimits="true"/>
  <option gravity="0 0 -9.81" timestep="0.002" integrator="implicitfast"/>

  <visual>
    <global offwidth="1280" offheight="960"/>
    <headlight diffuse="0.55 0.55 0.55" ambient="0.4 0.4 0.4" specular="0.1 0.1 0.1"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.35 0.45 0.58" rgb2="0.08 0.1 0.14"
             width="256" height="512"/>
    <texture type="2d" name="bench" builtin="checker" rgb1="0.32 0.33 0.35" rgb2="0.24 0.25 0.27"
             width="256" height="256"/>
    <material name="bench" texture="bench" texrepeat="8 8" texuniform="true" reflectance="0.05"/>
  </asset>

  <worldbody>
    <light pos="0.3 -0.3 1.2" dir="-0.2 0.25 -1" directional="true"/>
    <geom name="table" type="plane" size="1 1 0.05" material="bench"/>
    <camera name="overhead" pos="0.24 -0.52 0.46" xyaxes="1 0 0 0 0.64 0.77"/>

    <body name="upper" pos="0 0 0.34">
      <joint name="shoulder" type="hinge" axis="0 -1 0" range="-2.0 2.0"/>
      <geom name="upper_g" type="capsule" fromto="0 0 0 0.22 0 0" size="0.018" mass="0.4"
            rgba="0.72 0.73 0.76 1"/>
      <body name="fore" pos="0.22 0 0">
        <joint name="elbow" type="hinge" axis="0 -1 0" range="-2.4 2.4"/>
        <geom name="fore_g" type="capsule" fromto="0 0 0 0.20 0 0" size="0.015" mass="0.3"
              rgba="0.72 0.73 0.76 1"/>
        <body name="tip" pos="0.20 0 0">
          <geom name="pad" type="box" size="0.014 0.014 0.010" mass="0.05" rgba="0.88 0.58 0.18 1"/>
          <site name="pad_site" type="box" size="0.016 0.016 0.012" rgba="1 0 0 0"/>
          <site name="eye" pos="0.016 0 0" xyaxes="0 1 0 0 0 1" size="0.004" rgba="0 1 1 0.4"/>
          <camera name="wrist_cam" pos="-0.09 -0.055 0.05" xyaxes="0 -1 0 0.343 0 0.939" fovy="58"/>
        </body>
      </body>
    </body>

    <body name="cube" pos="0.30 0 0.02">
      <freejoint name="cube_free"/>
      <geom name="cube_g" type="box" size="0.02 0.02 0.02" mass="0.06" rgba="0.16 0.62 0.38 1"/>
    </body>
  </worldbody>

  <actuator>
    <position name="shoulder_act" joint="shoulder" kp="30" kv="3" ctrlrange="-2 2" forcerange="-12 12"/>
    <position name="elbow_act"    joint="elbow"    kp="20" kv="2" ctrlrange="-2.4 2.4" forcerange="-8 8"/>
  </actuator>

  <sensor>
    <jointpos name="shoulder_q"  joint="shoulder"/>
    <jointpos name="elbow_q"     joint="elbow"/>
    <jointvel name="shoulder_qd" joint="shoulder"/>
    <jointvel name="elbow_qd"    joint="elbow"/>
    <actuatorfrc name="shoulder_tau" actuator="shoulder_act"/>
    <actuatorfrc name="elbow_tau"    actuator="elbow_act"/>
    <framepos  name="pad_xyz"  objtype="site" objname="pad_site"/>
    <framequat name="pad_quat" objtype="site" objname="pad_site"/>
    <!-- TODO(you): add two more sensors here. Both take name= and site=, and
         both sites already exist in the body tree above.
           - a "touch" sensor named pad_touch on site pad_site
               reports the total normal force on that site's zone, in newtons
           - a "rangefinder" sensor named tip_range on site eye
               casts a ray along the site's z axis, reports metres, -1 on a miss
         Re-run afterwards: data.sensordata grows from 16 floats to 18. -->
    <framepos name="cube_xyz" objtype="body" objname="cube"/>
  </sensor>
</mujoco>
"""

L1, L2, BASE_Z = 0.22, 0.20, 0.34
CTRL_HZ, CAM_HZ = 50, 10


def ik(x, z):
    """Two-link inverse kinematics in the x-z plane, elbow-up branch (lesson 1.15)."""
    dx, dz = x, z - BASE_Z
    c2 = (dx * dx + dz * dz - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    th2 = -np.arccos(np.clip(c2, -1.0, 1.0))
    th1 = np.arctan2(dz, dx) - np.arctan2(L2 * np.sin(th2), L1 + L2 * np.cos(th2))
    return th1, th2


def read(data, name):
    """One sensor's slice of data.sensordata, by name.

    TODO(you): return np.array(data.sensor(name).data).

    data.sensor(name) is MuJoCo's named accessor; .data is a live view onto the
    slice of data.sensordata that this sensor owns. np.array copies it, so a
    logged value does not change under you on the next step.
    """
    raise NotImplementedError


def layout(model, data):
    print(f"{model.nsensor} sensors, {model.nsensordata} floats in data.sensordata\n")
    print(f"  {'name':13s} {'type':22s} {'adr':>4s} {'dim':>4s}   value")
    for i in range(model.nsensor):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        kind = mujoco.mjtSensor(model.sensor_type[i]).name.replace("mjSENS_", "").lower()
        adr, dim = model.sensor_adr[i], model.sensor_dim[i]
        print(f"  {name:13s} {kind:22s} {adr:4d} {dim:4d}   "
              f"{np.round(data.sensordata[adr:adr + dim], 4)}")
    print("\n  oracle, no sensor declared:")
    print(f"    data.body('cube').xpos   = {np.round(data.body('cube').xpos, 6)}")
    print(f"    sensor('cube_xyz')       = {np.round(read(data, 'cube_xyz'), 6)}")
    print("    identical: a framepos sensor relocates the truth, it does not measure it.\n")


def episode(model, data):
    """Descend onto the cube on three clocks: physics, control and camera."""
    steps_per_ctrl = int(round(1 / (CTRL_HZ * model.opt.timestep)))
    frame_dt, next_frame, frames = 1.0 / CAM_HZ, 0.0, 0
    renderer = mujoco.Renderer(model, height=240, width=320)
    log = []
    wall = time.perf_counter()
    for tick in range(int(3.0 * CTRL_HZ)):
        t = tick / CTRL_HZ
        z = 0.16 if t < 0.4 else max(0.055, 0.16 - 0.09 * (t - 0.4))
        data.ctrl[:] = ik(0.30, z)                       # 50 Hz: the controller
        for _ in range(steps_per_ctrl):
            mujoco.mj_step(model, data)                  # 500 Hz: the physics
            if data.time >= next_frame:                  # 10 Hz: the camera
                renderer.update_scene(data, camera="overhead")
                renderer.render()
                frames += 1
                next_frame += frame_dt
        log.append((data.time,
                    float(read(data, "shoulder_q")[0]),
                    float(read(data, "shoulder_tau")[0]),
                    float(read(data, "tip_range")[0]),
                    float(read(data, "pad_touch")[0]),
                    float(read(data, "pad_xyz")[2])))
    wall = time.perf_counter() - wall
    renderer.close()
    log = np.array(log)
    print(f"   {data.time:.1f} s simulated in {wall:.2f} s wall, "
          f"{frames} frames at {CAM_HZ} Hz\n")
    print("   t(s)  shoulder_q  shoulder_tau  tip_range  pad_touch   pad_z")
    for row in log[::20]:
        print("  {:5.2f}  {:10.4f}  {:12.4f}  {:9.4f}  {:9.3f}  {:6.4f}".format(*row))
    first = int(np.argmax(log[:, 4] > 0.05))
    print(f"\n   rangefinder at rest before the descent: {log[0, 3]:+.1f} "
          f"(-1 means the ray hit nothing)")
    print(f"   first contact at t = {log[first, 0]:.2f} s: pad_touch goes "
          f"{log[first - 1, 4]:.3f} -> {log[first, 4]:.3f} N in one 50 Hz tick")
    return log


def plot_trace(log, path=None):
    path = beside("sensor_trace.png", path)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 6), sharex=True)
    for ax, col, label, colour in ((axes[0], 5, "pad height (m)", "#2d5d7c"),
                                   (axes[1], 3, "rangefinder (m)", "#b07d22"),
                                   (axes[2], 4, "touch (N)", "#9c3b2e")):
        ax.plot(log[:, 0], log[:, col], lw=2, color=colour)
        ax.set_ylabel(label)
        ax.grid(color="#e8e7e3", lw=0.7)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"   wrote {path.name}")


def render_views(model, data, path=None):
    path = beside("sensor_views.png", path)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    renderer = mujoco.Renderer(model, height=360, width=480)
    panels = []
    for cam in ("overhead", "wrist_cam"):
        renderer.update_scene(data, camera=cam)
        panels.append((f"{cam} rgb  uint8", renderer.render(), None))
    # TODO(you): render the same overhead view as depth instead of colour.
    #   1. renderer.enable_depth_rendering()
    #   2. point the renderer at the overhead camera again and render
    #   3. renderer.disable_depth_rendering()  - the flag is sticky, and a
    #      renderer left in depth mode returns float32 metres where the next
    #      caller expects uint8 pixels
    # Assign the result to `depth`.
    depth = None
    raise NotImplementedError
    # Sky pixels come back at the far plane, tens of metres out. Left in, they
    # flatten every real distance to the bottom of the colour scale.
    near = depth[depth < 5.0]
    panels.append((f"overhead depth  float32 m, clipped at {near.max():.2f}",
                   np.clip(depth, near.min(), near.max()), "magma"))
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera="overhead")
    seg = renderer.render()[:, :, 0]
    renderer.disable_segmentation_rendering()
    panels.append(("overhead segmentation  int32 geom id", seg, "tab20"))
    renderer.close()

    print(f"   rgb   {panels[0][1].shape} {panels[0][1].dtype} "
          f"range {panels[0][1].min()}-{panels[0][1].max()}")
    print(f"   depth {depth.shape} {depth.dtype} "
          f"range {depth.min():.3f}-{depth.max():.3f} m "
          f"({int((depth >= 5).sum())} pixels past 5 m: that is the sky)")
    print(f"   seg   {seg.shape} {seg.dtype} geom ids {np.unique(seg)}")

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for ax, (title, img, cmap) in zip(axes.ravel(), panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"   wrote {path.name}")


def timing(model, data, n=200):
    """Medians, not means: a laptop under thermal management produces outliers."""
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        for _ in range(50):
            mujoco.mj_step(model, data)
        samples.append((time.perf_counter() - t0) / 50 * 1e3)
    step_ms = statistics.median(samples)
    print(f"   mj_step                {step_ms * 1000:8.1f} us   (median of {n})")
    for w, h in ((320, 240), (480, 360), (640, 480)):
        renderer = mujoco.Renderer(model, height=h, width=w)
        renderer.update_scene(data, camera="overhead")
        renderer.render()                       # warm the GL context
        samples = []
        for _ in range(n):
            t0 = time.perf_counter()
            renderer.update_scene(data, camera="overhead")
            renderer.render()
            samples.append((time.perf_counter() - t0) * 1e3)
        renderer.close()
        frame_ms = statistics.median(samples)
        print(f"   render {w}x{h}          {frame_ms:8.2f} ms   = {frame_ms / step_ms:6.0f} steps")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    if which in ("layout", "all"):
        print("A. What data.sensordata holds")
        layout(model, data)
    if which in ("episode", "render", "timing", "all"):
        print("B. One descent onto the cube")
        log = episode(model, data)
        plot_trace(log)
    if which in ("render", "all"):
        print("\nC. What the cameras return")
        render_views(model, data)
    if which in ("timing", "all"):
        print("\nD. What rendering costs")
        timing(model, data)


if __name__ == "__main__":
    main()
