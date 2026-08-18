"""Solution - Lesson 2.3. The interactive viewer, with your loop in charge.

macOS:            mjpython watch_it_fall.py [seconds]
Linux / Windows:  python   watch_it_fall.py [seconds]

Once the window is up: drag to orbit, scroll to zoom, double-click a body to select it,
then ctrl-drag to shove it around while the physics keeps running.
"""
import sys
import time

import mujoco
import mujoco.viewer

SCENE_XML = """
<mujoco model="drop-test">
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 0.05" rgba="0.92 0.90 0.85 1"/>
    <geom name="ramp" type="box" size="0.35 0.25 0.02" pos="0.15 0 0.18"
          euler="0 -20 0" rgba="0.61 0.23 0.18 1"/>
    <body name="ball" pos="-0.05 0 1.0">
      <freejoint/>
      <geom name="ball" type="sphere" size="0.05" rgba="0.18 0.36 0.49 1"/>
    </body>
  </worldbody>
</mujoco>
"""

RESET_AFTER = 3.0  # simulated seconds before the ball goes back up top


def main(seconds=60.0):
    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        wall_start = time.time()
        while viewer.is_running() and time.time() - wall_start < seconds:
            step_start = time.time()

            mujoco.mj_step(model, data)
            if data.time > RESET_AFTER:
                mujoco.mj_resetData(model, data)

            viewer.sync()

            # Pace the loop to wall-clock time. Without this the sim runs as fast as
            # the CPU allows, which for this scene is far faster than real time.
            slack = model.opt.timestep - (time.time() - step_start)
            if slack > 0:
                time.sleep(slack)

    print(f"viewer closed after {data.time:.2f} simulated seconds")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 60.0)
