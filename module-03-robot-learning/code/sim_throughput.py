"""Lesson 3.16 - how much simulation a locomotion run needs, and how much your
laptop can supply.

Builds a 12-degree-of-freedom quadruped in MuJoCo (four legs, three joints
each, floating trunk, contact with the floor), steps it as fast as one CPU core
can, then steps it again with a locomotion-sized policy network in the loop.
Then you turn that into wall-clock for a real training budget.

Needs `mujoco`, which Module 2 installed. On Intel macOS pin it:
    pip install "mujoco<3.11"

Run:  python sim_throughput.py
About half a minute.
"""
import time

import mujoco
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)      # one core, so the comparison is per-core

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
    """TODO(you): hours needed to run `steps` at `steps_per_second`. One line."""
    raise NotImplementedError


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

    # TODO(you): finish the arithmetic.
    #   1. hours for BUDGET_STEPS on this machine, using `with_policy`
    #      (physics alone is not what training costs)
    #   2. the same for every entry in REPORTED, and the ratio
    #   3. multiply by RUNS_PER_RESULT - one run is never the real cost
    #   4. price the GPU version at A100_USD_PER_HOUR
    # Print all four. Then write one sentence in a comment: which of these
    # numbers is the reason you rent a GPU, and which one is a red herring?
    raise NotImplementedError


if __name__ == "__main__":
    main()
