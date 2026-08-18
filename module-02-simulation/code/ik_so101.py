"""Lesson 2.9 - your Module 1 damped-least-squares IK, driving the SO-101, and
the point where agreement with the physics stops.

Imports the forward kinematics from fk_so101.py, so finish that one first.

Run:  python ik_so101.py              (Jacobian cross-check + 200-target IK eval)
      python ik_so101.py --settle     (command each solution, let the physics run)
"""
import sys

import numpy as np
import mujoco

from fk_so101 import EE_SITE, chain_geometry, fk, load

LAMBDA = 0.05      # damping; Module 1 called it λ
ALPHA = 0.6        # step size
TOL = 1e-4         # 0.1 mm counts as arrived


def jacobian_fd(q, offsets, site_offset, eps=1e-6):
    """3x5 position Jacobian by finite differences on your own FK.

    Column j answers: if joint j moves one radian, which way does the tool tip
    go, and how fast?
    """
    here = fk(q, offsets, site_offset)[:3, 3]
    # TODO(you): for each joint j, nudge q[j] by eps, run fk again, and take
    #            (moved - here) / eps as column j. np.column_stack the five.
    raise NotImplementedError


def jacobian_mujoco(model, data, q, site_id):
    """MuJoCo's analytic Jacobian for the same site, for comparison."""
    data.qpos[:5] = q
    data.qpos[5] = 0.0
    mujoco.mj_forward(model, data)
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, site_id)
    return jacp[:, :5]


def ik_step(target, q, offsets, site_offset, lo, hi, lam=LAMBDA, alpha=ALPHA):
    """One damped-least-squares update, clamped to the joint limits."""
    error = target - fk(q, offsets, site_offset)[:3, 3]
    J = jacobian_fd(q, offsets, site_offset)
    # TODO(you): the Lesson 1.15 update, now with a 3x5 Jacobian:
    #              dq = Jᵀ (J Jᵀ + λ² I)⁻¹ e        (I is 3x3 here)
    #            then step by alpha and clip into [lo, hi].
    #            Return (new_q, float(np.linalg.norm(error))).
    raise NotImplementedError


def ik_from(target, q0, offsets, site_offset, lo, hi, iters=300):
    q = np.array(q0, dtype=float)
    for k in range(iters):
        q, err = ik_step(target, q, offsets, site_offset, lo, hi)
        if err < TOL:
            return q, True, k
    return q, False, iters


def ik(target, offsets, site_offset, lo, hi, restarts=5, seed=0):
    """DLS with random restarts. IK is a local method; a restart is the cheapest
    escape from a local minimum."""
    rng = np.random.default_rng(seed)
    q0, total = np.zeros(5), 0
    for _ in range(restarts):
        q, converged, k = ik_from(target, q0, offsets, site_offset, lo, hi)
        total += k
        if converged:
            return q, True, total
        q0 = rng.uniform(lo, hi)
    return q, False, total


def check_jacobians(model, data, offsets, site_offset, site_id, lo, hi, n=500):
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(n):
        q = rng.uniform(lo, hi)
        mine = jacobian_fd(q, offsets, site_offset)
        theirs = jacobian_mujoco(model, data, q, site_id)
        worst = max(worst, float(np.abs(mine - theirs).max()))
    print(f"{n} configurations: worst Jacobian element disagreement {worst:.3e}")
    print("  (finite differences on your FK vs mj_jacSite)")
    assert worst < 1e-5


def eval_ik(offsets, site_offset, lo, hi, n=200):
    """Sample reachable targets by running FK on random joint angles, then ask
    IK to find its way back."""
    rng = np.random.default_rng(2)
    solved, iters, residuals, cases, underground = 0, [], [], [], 0
    for i in range(n):
        q_true = rng.uniform(0.8 * lo, 0.8 * hi)
        target = fk(q_true, offsets, site_offset)[:3, 3]
        underground += target[2] < 0.0
        q, converged, k = ik(target, offsets, site_offset, lo, hi, seed=i)
        solved += converged
        iters.append(k)
        residuals.append(float(np.linalg.norm(fk(q, offsets, site_offset)[:3, 3] - target)))
        if converged:
            cases.append((target, q))
    print(f"\n{solved}/{n} targets solved; median {int(np.median(iters))} iterations; "
          f"median residual {np.median(residuals):.2e} m")
    print(f"  {underground}/{n} of those targets are below the floor: joint limits "
          "describe configurations, not sensible places to be.")
    return cases


def settle(model, data, cases, steps=400):
    """Command each IK answer as a position setpoint and run the physics."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    free, touching = [], []
    for target, q in cases:
        mujoco.mj_resetData(model, data)
        data.qpos[:5] = q
        data.ctrl[:5] = q
        data.ctrl[5] = 0.0
        for _ in range(steps):
            mujoco.mj_step(model, data)
        gap = float(np.linalg.norm(data.site_xpos[site_id] - target))
        (touching if data.ncon > 0 else free).append(gap)
    f = np.array(free)
    print(f"\nafter {steps * model.opt.timestep:.0f} s of physics from the IK answer:")
    print(f"  {len(f)} poses in free space   median gap {np.median(f) * 1000:.3f} mm, "
          f"worst {f.max() * 1000:.3f} mm")
    if touching:
        t = np.array(touching)
        print(f"  {len(t)} poses in contact     median gap {np.median(t) * 1000:.1f} mm, "
              f"worst {t.max() * 1000:.1f} mm")
        print("  the second group is the arm pressed into the floor: kinematics has no "
              "opinion about solid objects.")


def main():
    model, data = load()
    offsets, site_offset = chain_geometry(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    lo, hi = model.jnt_range[:5, 0], model.jnt_range[:5, 1]

    check_jacobians(model, data, offsets, site_offset, site_id, lo, hi)
    cases = eval_ik(offsets, site_offset, lo, hi)
    if "--settle" in sys.argv:
        settle(model, data, cases)


if __name__ == "__main__":
    main()
