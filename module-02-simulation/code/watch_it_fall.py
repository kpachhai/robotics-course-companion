"""Lesson 2.3 - the interactive viewer, with your loop in charge.

macOS:            mjpython watch_it_fall.py [seconds]
Linux / Windows:  python   watch_it_fall.py [seconds]

Run it with plain `python` on a Mac first, on purpose, and read the error you get.
Recognising that message later is worth the ten seconds it costs now.
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
    """Open the passive viewer and drive the physics yourself.

    Skeleton to fill in:

        model = mujoco.MjModel.from_xml_string(SCENE_XML)
        data  = mujoco.MjData(model)

        with mujoco.viewer.launch_passive(model, data) as viewer:
            wall_start = time.time()
            while viewer.is_running() and time.time() - wall_start < seconds:
                step_start = time.time()
                mujoco.mj_step(model, data)                 # advance one timestep
                if data.time > RESET_AFTER:
                    mujoco.mj_resetData(model, data)        # send the ball back up
                viewer.sync()                               # push state to the window
                slack = model.opt.timestep - (time.time() - step_start)
                if slack > 0:
                    time.sleep(slack)                       # pace to wall-clock time

    Delete the sleep once it works and watch what happens. That is the difference
    between "real time" and "as fast as this laptop can go".
    """
    # TODO(you)
    raise NotImplementedError


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 60.0)
