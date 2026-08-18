"""Lesson 2.16 - randomise the scene three ways and measure which ones cost you anything.

Run:  python randomize.py            all three sweeps  (about 5 minutes)
      python randomize.py --full     n = 20 per row instead of 10
"""
import sys

import mujoco
import numpy as np

import so101_pick as sp

NOMINAL = np.array([0.22, 0.0])


def randomised_episode(rng, width=0.0, yaw=0.0, mass_range=None, mu_range=None,
                       gain_scale=None, visual=False, trace=False):
    arm = sp.Arm()
    model = arm.model
    if trace:
        arm.trace = []
    if mass_range is not None:
        model.body_mass[model.body("cube").id] = rng.uniform(*mass_range)
    if mu_range is not None:
        model.geom_friction[model.geom_priority == 1, 0] = rng.uniform(*mu_range)
    if gain_scale is not None:
        s = rng.uniform(*gain_scale, size=6)
        model.actuator_gainprm[:6, 0] *= s               # kp
        model.actuator_biasprm[:6, 1] *= s               # -kp, must track it
    if visual:
        randomise_appearance(model, rng)
    xy = NOMINAL + rng.uniform(-width, width, size=2)
    arm.place_cube(xy[0], xy[1])
    if yaw:
        angle = rng.uniform(-yaw, yaw)
        arm.data.qpos[9:13] = (np.cos(angle / 2), 0, 0, np.sin(angle / 2))
        mujoco.mj_forward(model, arm.data)
    ok = sp.pick_and_place(arm)["success"]
    return (ok, np.array(arm.trace)) if trace else ok


def randomise_appearance(model, rng):
    """Everything a camera sees and no joint feels."""
    for name in ("cube_mat", "table_mat", "bin_mat"):
        mid = model.material(name).id
        model.mat_rgba[mid, :3] = rng.uniform(0.15, 0.9, size=3)
    lid = model.light("key").id
    model.light_pos[lid] = (rng.uniform(-0.4, 0.6), rng.uniform(-0.5, 0.5), rng.uniform(0.7, 1.6))
    model.light_diffuse[lid] = rng.uniform(0.4, 1.0, size=3)
    cid = model.camera("overhead").id
    model.cam_pos[cid] += rng.uniform(-0.04, 0.04, size=3)


def position_sweep(n=12, seed=7):
    print("cube position randomised in a square of half-width w around the nominal spot")
    for width in (0.00, 0.04, 0.08, 0.12, 0.16):
        rng = np.random.default_rng(seed)
        ok = sum(randomised_episode(rng, width=width) for _ in range(n))
        print(f"  w = {width * 100:4.1f} cm   success {ok:2d}/{n}")


def physics_sweep(n=12, seed=7, width=0.04):
    print(f"\nphysics randomised, cube position fixed at w = {width * 100:.0f} cm")
    rows = [("nothing (baseline)", {}),
            ("cube mass 30-1200 g", dict(mass_range=(0.03, 1.20))),
            ("finger friction 0.1-1.4", dict(mu_range=(0.1, 1.4))),
            ("servo gains x0.2-4.0", dict(gain_scale=(0.2, 4.0))),
            ("all three at once", dict(mass_range=(0.03, 1.20), mu_range=(0.1, 1.4),
                                       gain_scale=(0.2, 4.0)))]
    for label, kw in rows:
        rng = np.random.default_rng(seed)
        ok = sum(randomised_episode(rng, width=width, **kw) for _ in range(n))
        print(f"  {label:26s} success {ok:2d}/{n}")


def visual_sweep(n=12, seed=7, width=0.04):
    print(f"\nappearance randomised, physics untouched")
    rng = np.random.default_rng(seed)
    ok = sum(randomised_episode(rng, width=width, visual=True) for _ in range(n))
    print(f"  colours, light and camera   success {ok:2d}/{n}")

    arm = sp.Arm()
    renderer = mujoco.Renderer(arm.model, 240, 320)
    renderer.update_scene(arm.data, camera="overhead")
    base = renderer.render().astype(float)
    diffs = []
    rng = np.random.default_rng(seed)
    for _ in range(5):
        arm2 = sp.Arm()
        randomise_appearance(arm2.model, rng)
        r2 = mujoco.Renderer(arm2.model, 240, 320)
        r2.update_scene(arm2.data, camera="overhead")
        diffs.append(np.abs(r2.render().astype(float) - base).mean())
    print(f"  mean per-pixel change vs the nominal render: "
          f"{np.mean(diffs):.1f} of 255 (min {min(diffs):.1f}, max {max(diffs):.1f})")


def data_spread(n=8, seed=11):
    """Success is the label. The joint trace is the data. They move separately."""
    print("\nspread of the recorded joint trajectory across episodes, in degrees")
    families = [("appearance only", dict(width=0.0, visual=True)),
                ("dynamics only", dict(width=0.0, mass_range=(0.03, 1.20),
                                       mu_range=(0.1, 1.4), gain_scale=(0.2, 4.0))),
                ("cube pose, w = 4 cm", dict(width=0.04))]
    for label, kw in families:
        rng = np.random.default_rng(seed)
        traces = [randomised_episode(rng, trace=True, **kw)[1] for _ in range(n)]
        stack = np.stack([t[:min(map(len, traces))] for t in traces])
        spread = np.degrees(stack.std(axis=0)).mean()
        print(f"  {label:22s} mean per-joint spread {spread:6.3f} deg")


if __name__ == "__main__":
    n = 20 if "--full" in sys.argv else 10
    position_sweep(n)
    physics_sweep(n)
    visual_sweep(n)
    data_spread()
