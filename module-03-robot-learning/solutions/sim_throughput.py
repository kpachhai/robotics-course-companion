"""Solution - Lesson 3.16: what a locomotion training run costs, measured.

Builds a 12-degree-of-freedom quadruped in MuJoCo, measures how fast one CPU
core can step it with and without a policy in the loop, and turns that into
wall-clock and dollars for a real training budget.

Run:  python sim_throughput.py

Measured on a 2018 4-core Intel laptop under load. Re-measure on your own
machine; the exercise is the measurement, not my number.
"""
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)      # one core, so the comparison is per-core

# The lesson quotes these two numbers in prose, so the figure has to show the
# same ones. It cannot re-measure at build time: whoever last ran generate.py
# would silently rewrite a stated measurement with their own hardware's, and
# the figure and the paragraph under it would disagree. Same reason
# chunking_sweep.py keeps a committed record.
MEASUREMENT = Path(__file__).with_name("sim_throughput_measured.json")

LEG = """
      <body name="{name}_hip" pos="{x} {y} 0">
        <joint name="{name}_abduct" type="hinge" axis="1 0 0" range="-0.8 0.8"/>
        <geom type="capsule" fromto="0 0 0 0 {span} 0" size="0.022"/>
        <body name="{name}_thigh" pos="0 {span} 0">
          <joint name="{name}_hip_pitch" type="hinge" axis="0 1 0" range="-1.6 1.6"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.16" size="0.020"/>
          <body name="{name}_calf" pos="0 0 -0.16">
            <joint name="{name}_knee" type="hinge" axis="0 1 0" range="-2.4 -0.2"/>
            <geom type="capsule" fromto="0 0 0 0 0 -0.16" size="0.016"/>
            <geom name="{name}_foot" type="sphere" pos="0 0 -0.17" size="0.022"/>
          </body>
        </body>
      </body>
"""

SCENE = """
<mujoco model="quad12">
  <option timestep="0.004" iterations="10" ls_iterations="10" solver="Newton"/>
  <default>
    <geom friction="0.8 0.02 0.01" density="800"/>
    <joint damping="0.5" armature="0.01"/>
    <motor ctrlrange="-20 20"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="trunk" pos="0 0 0.35">
      <freejoint/>
      <geom type="box" size="0.20 0.09 0.05"/>
{legs}    </body>
  </worldbody>
  <actuator>
    {motors}
  </actuator>
</mujoco>
"""

CORNERS = [("fl", 0.17, 0.07), ("fr", 0.17, -0.07),
           ("hl", -0.17, 0.07), ("hr", -0.17, -0.07)]


def build_model():
    legs, motors = [], []
    for name, x, y in CORNERS:
        legs.append(LEG.format(name=name, x=x, y=y, span=0.055 if y > 0 else -0.055))
        motors += [f'<motor name="{name}_{j}" joint="{name}_{j}"/>'
                   for j in ("abduct", "hip_pitch", "knee")]
    return mujoco.MjModel.from_xml_string(
        SCENE.format(legs="".join(legs), motors="\n    ".join(motors)))


def locomotion_policy(n_obs=48, n_act=12):
    """The shape the locomotion literature actually uses: a small MLP, not a
    transformer. Nothing about legged RL needs a large network."""
    return nn.Sequential(nn.Linear(n_obs, 512), nn.ELU(),
                         nn.Linear(512, 256), nn.ELU(),
                         nn.Linear(256, 128), nn.ELU(),
                         nn.Linear(128, n_act))


def measure(model, policy=None, chunk=3000, trials=7, seed=0):
    """Steps per second, one environment, one core, optionally with a policy
    forward pass per step.

    Reports the BEST of several trials rather than the mean: a laptop under
    load produces a distribution with a long slow tail, and the fastest run is
    the closest estimate of what the machine can actually do.
    """
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    obs = torch.zeros(1, 48)
    for _ in range(500):                        # let it settle onto the floor
        mujoco.mj_step(model, data)
    best = np.inf
    for _ in range(trials):
        ctrl = rng.uniform(-4, 4, size=(chunk, model.nu))
        start = time.perf_counter()
        for k in range(chunk):
            if policy is None:
                data.ctrl[:] = ctrl[k]
            else:
                with torch.no_grad():
                    data.ctrl[:] = policy(obs).numpy()[0]
            mujoco.mj_step(model, data)
        best = min(best, time.perf_counter() - start)
    return chunk / best


def load_or_measure(model=None, **kw):
    """The committed measurement, or a fresh one if the record is missing.

    Used by the figure generator so the plotted numbers stay the ones the
    lesson quotes. Running this file directly always measures; only the figure
    reads the record.
    """
    if MEASUREMENT.exists():
        return json.loads(MEASUREMENT.read_text())
    model = model if model is not None else build_model()
    res = {"physics_only": measure(model, **kw),
           "with_policy": measure(model, policy=locomotion_policy(), **kw)}
    MEASUREMENT.write_text(json.dumps(res, indent=2) + "\n")
    return res


# ------------------------------------------------------------- the arithmetic
BUDGET_STEPS = 100_000_000          # a typical quadruped PPO run
RUNS_PER_RESULT = 20                # reward tuning, randomisation ranges, seeds
A100_USD_PER_HOUR = 1.19            # RunPod community, checked 2026-08-09
REPORTED = {
    # MuJoCo Playground, arXiv:2502.08844, Table VII, Go1JoystickFlatTerrain.
    # The authors' number on an A100. Not reproduced here.
    "MJX on one A100 (reported)": 417_451,
}


def wall_clock_hours(steps, steps_per_second):
    return steps / steps_per_second / 3600.0


def main():
    model = build_model()
    print(f"model: {model.nu} actuated joints, {model.nv} velocity coordinates, "
          f"{model.ngeom} geoms, {model.opt.timestep * 1000:.0f} ms timestep\n")

    physics_only = measure(model)
    with_policy = measure(model, policy=locomotion_policy())
    print(f"physics alone        {physics_only:9,.0f} steps/s  "
          f"({physics_only * model.opt.timestep:,.0f}x real time)")
    print(f"physics + policy     {with_policy:9,.0f} steps/s  "
          f"({with_policy * model.opt.timestep:,.0f}x real time)")

    here = wall_clock_hours(BUDGET_STEPS, with_policy)
    print(f"\none run of {BUDGET_STEPS:,} steps")
    print(f"  this machine                 {here:8,.1f} h   ({here / 24:5.1f} days)")
    for label, rate in REPORTED.items():
        print(f"  {label:<28} {wall_clock_hours(BUDGET_STEPS, rate):8,.1f} h   "
              f"({rate / with_policy:.0f}x this machine)")

    print(f"\n{RUNS_PER_RESULT} runs, a realistic count once you have tuned the "
          f"reward and the randomisation ranges")
    print(f"  this machine                 {here * RUNS_PER_RESULT / 24:8,.1f} days")
    for label, rate in REPORTED.items():
        hours = wall_clock_hours(BUDGET_STEPS, rate) * RUNS_PER_RESULT
        # Re-check the rate before quoting it: GPU rental prices move.
        print(f"  {label:<28} {hours:8,.1f} h   "
              f"= ${hours * A100_USD_PER_HOUR:,.2f} rented")

    # The red herring is the single-run number: two or three days is annoying,
    # not disqualifying. The real reason to rent is the multiplier - twenty runs
    # is months on this machine and an afternoon on a rented GPU - plus the fact
    # that MuJoCo Playground's GPU path (JAX) has no Apple GPU backend at all.

    assert with_policy > 200, "something is wrong: this model should step far faster"
    assert with_policy < physics_only, "the policy forward pass cannot be free"


if __name__ == "__main__":
    main()
