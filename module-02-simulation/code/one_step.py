"""Lesson 2.2 - one physics step, written out by hand.

Four experiments, numpy only. Together they are a physics engine's whole job:
  1. integrate            - explicit vs semi-implicit Euler, one line apart
  2. contact as a force   - the penalty spring, and the stiffness where it explodes
  3. contact as a constraint - solve for the impulse instead; survives a 50 ms step
  4. two contacts at once - why the solver sweeps the rows more than once

Experiment 4 is written for you. Read it; it is the shape of a real solver.

Run:  python one_step.py
"""
import numpy as np

G = 9.81


# ---------------------------------------------------------------- 1. integrate
def spring(x0=1.0, v0=0.0, k=100.0, m=1.0, dt=0.01, steps=2000, semi_implicit=True):
    """Mass on a spring. Returns (energy_start, energy_end).

    TODO(you): inside the loop, after computing `v_new`, decide which velocity
    the position update uses.
      explicit Euler    -> the velocity you started the step with  (v)
      semi-implicit     -> the velocity you just computed          (v_new)
    That choice is the whole experiment. One variable.
    """
    x, v = x0, v0
    energy = lambda x, v: 0.5 * m * v * v + 0.5 * k * x * x   # noqa: E731
    e0 = energy(x, v)
    for _ in range(steps):
        a = -k * x / m
        v_new = v + a * dt
        v_used = ...                                 # TODO(you)
        if v_used is Ellipsis:
            raise NotImplementedError("set v_used to v (explicit) or v_new (semi-implicit)")
        x = x + v_used * dt
        v = v_new
    return e0, energy(x, v)


# ------------------------------------------------- 2. contact as a stiff force
def drop_penalty(k, z0=0.30, m=1.0, dt=0.002, steps=1500, damp_ratio=0.1):
    """Drop a ball; a stiff spring under the floor pushes it back out.

    Returns (deepest_penetration_m, diverged).

    TODO(you): when the ball is below the floor (z < 0) add the penalty force
    to `f`: a spring term -k*z (z is negative, so this pushes up) and a damping
    term -c*v. Leave `f` alone when z >= 0; nothing is touching.
    """
    c = damp_ratio * 2.0 * np.sqrt(k * m)
    z, v, deepest = z0, 0.0, 0.0
    for _ in range(steps):
        f = -m * G
        # TODO(you): the contact force
        v = v + (f / m) * dt
        z = z + v * dt
        deepest = max(deepest, -z)
        if not np.isfinite(z) or abs(z) > 10.0:
            return deepest, True
    return deepest, False


# ------------------------------------------------ 3. contact as a constraint
def drop_solved(dt, z0=0.30, m=1.0, sim_time=3.0, beta=0.2, softness=0.5):
    """Same drop. No stiff spring: solve for the impulse that stops the ball.

    Returns (resting_penetration_m, diverged).

    TODO(you): fill in the branch taken when the step would end illegally.
      depth    = how far into the floor the ball already is, never negative
      v_target = beta * depth / dt      the separation speed we will allow
      lam      = the impulse that turns v_free into v_target, divided by
                 (1 + softness), and clamped at zero because a contact can
                 push and never pull:  max(0, m*(v_target - v_free)/(1+softness))
      v        = v_free + lam/m
    """
    z, v = z0, 0.0
    steps = int(sim_time / dt)
    for _ in range(steps):
        v_free = v - G * dt
        if z + v_free * dt < 0.0:                    # this step would end illegal
            raise NotImplementedError                # TODO(you)
        else:
            v = v_free
        z = z + v * dt
        if not np.isfinite(z) or abs(z) > 10.0:
            return float("nan"), True
    return -z, False


# ------------------------------------- 4. two contacts that have to agree
def stack_solved(sweeps, dt=0.002, sim_time=2.0, r=0.05, m=1.0, beta=0.2, softness=0.5):
    """A ball resting on a ball resting on the floor: two coupled contacts.

    Provided, not a TODO. `sweeps` is how many passes the solver makes over the
    two rows per step. Each row is experiment 3 again; the only new thing is
    that row 1 changes a velocity row 0 already fixed, so one pass is not enough.
    Returns the resting penetration of the lower contact, in metres.
    """
    z = np.array([r, 3.0 * r])
    v = np.zeros(2)
    for _ in range(int(sim_time / dt)):
        v = v - G * dt
        lam = np.zeros(2)                            # accumulated impulses, >= 0
        for _ in range(sweeps):
            # row 0: floor against the lower ball
            depth = max(0.0, r - z[0])
            dlam = m * (beta * depth / dt - v[0]) / (1.0 + softness)
            dlam = max(dlam, -lam[0])                # keep the total pushing, never pulling
            lam[0] += dlam
            v[0] += dlam / m
            # row 1: lower ball against upper ball
            depth = max(0.0, 2.0 * r - (z[1] - z[0]))
            m_eff = 1.0 / (1.0 / m + 1.0 / m)
            dlam = m_eff * (beta * depth / dt - (v[1] - v[0])) / (1.0 + softness)
            dlam = max(dlam, -lam[1])
            lam[1] += dlam
            v[1] += dlam / m
            v[0] -= dlam / m
        z = z + v * dt
    return r - z[0]


def main():
    print("1. the order of two lines (mass-spring, 20 s, dt = 0.01)")
    for name, semi in (("explicit Euler ", False), ("semi-implicit  ", True)):
        e0, e1 = spring(semi_implicit=semi)
        print(f"   {name}  energy {e0:6.1f} -> {e1:10.1f}   ({e1 / e0:,.2f}x)")

    print("\n2. contact as a stiff force (drop from 0.30 m, dt = 0.002)")
    print("   stiffness k      deepest penetration   stable?")
    for k in (1e3, 1e4, 1e5, 1e6, 1e7):
        deepest, diverged = drop_penalty(k)
        depth = "        -    " if diverged else f"{deepest * 1000:9.2f} mm"
        print(f"   {k:>10,.0f}      {depth}         {'no' if diverged else 'yes'}")
    target = 1e-4
    k_needed = 2 * G * 0.30 / target ** 2
    print(f"   for {target * 1000:.1f} mm you need k ~ {k_needed:,.0f}, "
          f"which needs dt < {2 / np.sqrt(k_needed) * 1e6:,.0f} us")

    print("\n3. contact as a constraint (same drop, solved)")
    print("   timestep     resting penetration   stable?")
    for dt in (0.002, 0.01, 0.02, 0.05):
        depth, diverged = drop_solved(dt)
        print(f"   {dt * 1000:5.0f} ms     {depth * 1000:9.3f} mm         "
              f"{'no' if diverged else 'yes'}")

    print("\n4. two contacts that have to agree (ball on ball on floor)")
    for sweeps in (1, 2, 5, 20):
        print(f"   {sweeps:2d} solver sweep(s)   lower contact "
              f"{stack_solved(sweeps) * 1e6:7.1f} um deep")


if __name__ == "__main__":
    main()
