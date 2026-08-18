"""Lesson 3.17 - tune a controller on one simulator, then deploy it on a
different one.

The plant is the gravity-loaded joint from Module 1: a pendulum held at
horizontal by a PD controller. Three parameters make it a *family* of plants
rather than one plant - link mass, joint damping, and how many steps late the
measurement arrives.

Two controllers get tuned on exactly the same grid of gains:
  - one on the nominal plant only
  - one on the average over a randomised family

Then both are deployed across a sweep of real plants and scored.

Everything is vectorised over plants: one simulation loop runs every
(gain pair x plant) combination at once, so the whole thing is seconds.

Run:  python domain_randomization.py
"""
import time

import numpy as np

DT, T_END, TARGET, TAU_MAX, LINK = 0.005, 4.0, np.pi / 2, 8.0, 0.4
N_STEPS = int(T_END / DT)
SCORE_FROM = int(2.0 / DT)          # score the last two seconds: settled, not rising
MAX_DELAY = 20                       # measurement-buffer depth, in control steps
NOMINAL = dict(mass=0.5, damping=0.05, delay=0)


def simulate(kp, kd, mass, damping, delay):
    """RMS settling error, in radians. Every argument is an array of length K;
    column i is one (controller, plant) pair, simulated independently."""
    kp, kd, mass, damping = map(np.asarray, (kp, kd, mass, damping))
    delay = np.asarray(delay, dtype=int)
    n = np.broadcast(kp, kd, mass, damping, delay).shape[0]
    theta = np.zeros(n)
    theta_dot = np.zeros(n)
    hist_th = np.zeros((MAX_DELAY + 1, n))
    hist_thd = np.zeros((MAX_DELAY + 1, n))
    cols = np.arange(n)
    squared = np.zeros(n)

    for k in range(N_STEPS):
        hist_th[k % (MAX_DELAY + 1)] = theta            # ring buffer of measurements
        hist_thd[k % (MAX_DELAY + 1)] = theta_dot
        old = (k - delay) % (MAX_DELAY + 1)
        th_seen, thd_seen = hist_th[old, cols], hist_thd[old, cols]

        torque = np.clip(kp * (TARGET - th_seen) - kd * thd_seen, -TAU_MAX, TAU_MAX)
        accel = ((torque - mass * 9.81 * LINK * np.sin(theta) - damping * theta_dot)
                 / (mass * LINK ** 2))
        theta_dot = theta_dot + accel * DT
        theta = theta + theta_dot * DT
        if k >= SCORE_FROM:
            squared += np.minimum((TARGET - theta) ** 2, 4.0)   # cap a diverged run
    return np.sqrt(squared / (N_STEPS - SCORE_FROM))


KP_GRID = np.linspace(2, 80, 27)
KD_GRID = np.linspace(0.0, 8.0, 21)


def tune(plants):
    """Best (kp, kd) by mean cost over `plants`, a list of parameter dicts."""
    kp, kd = (a.ravel() for a in np.meshgrid(KP_GRID, KD_GRID, indexing="ij"))
    total = np.zeros(len(kp))
    for p in plants:
        total += simulate(kp, kd, np.full(len(kp), p["mass"]),
                          np.full(len(kp), p["damping"]),
                          np.full(len(kp), p["delay"], int))
    best = int(np.argmin(total))
    return float(total[best] / len(plants)), (float(kp[best]), float(kd[best]))


def score(gains, plant):
    """Cost of one controller on one plant."""
    return float(simulate(*(np.array([g]) for g in gains),
                          np.array([plant["mass"]]), np.array([plant["damping"]]),
                          np.array([plant["delay"]]))[0])


# ---------------------------------------------------------- what to randomise
def randomised_family(rng, n=16):
    """TODO(you): return `n` plant dicts drawn from ranges wide enough that the
    real robot is plausibly one of them.

    Suggested ranges, and the reasoning is the exercise:
        mass     uniform on [0.3, 1.0]     you measured 0.5 and you were wrong
        damping  uniform on [0.0, 0.4]     grease, wear, temperature
        delay    integer on 0..12 steps    0 to 60 ms of sensing and comms lag

    Pick them yourself and defend the widths. Too narrow and reality falls
    outside the family; too wide and you are training for plants that will
    never exist, which costs performance on the one that will.
    """
    raise NotImplementedError


def deployment_sweep(gains, delays):
    """TODO(you): score `gains` on the nominal plant at each delay in `delays`.
    Return a list of costs. Two lines."""
    raise NotImplementedError


def main():
    started = time.perf_counter()
    cost_nominal, gains_nominal = tune([NOMINAL])
    print(f"tuned on the nominal plant only:  kp={gains_nominal[0]:.0f} "
          f"kd={gains_nominal[1]:.2f}   cost there = {cost_nominal:.4f} rad")

    rng = np.random.default_rng(0)
    family = randomised_family(rng)
    cost_family, gains_dr = tune(family)
    print(f"tuned on {len(family)} randomised plants:      kp={gains_dr[0]:.0f} "
          f"kd={gains_dr[1]:.2f}   mean cost = {cost_family:.4f} rad")

    print(f"\nnominal-tuned controller on that same family: "
          f"{np.mean([score(gains_nominal, p) for p in family]):.4f} rad")
    print(f"randomised-tuned controller at nominal:       "
          f"{score(gains_dr, NOMINAL):.4f} rad")

    delays = list(range(0, 17, 2))
    print("\nlatency sweep (mass and damping held at nominal)")
    print(f"{'delay':>8}{'nominal-tuned':>16}{'randomised-tuned':>19}")
    for delay, a, b in zip(delays, deployment_sweep(gains_nominal, delays),
                           deployment_sweep(gains_dr, delays)):
        print(f"{delay * DT * 1000:6.0f}ms{a:16.3f}{b:19.3f}")

    print(f"\n{time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
