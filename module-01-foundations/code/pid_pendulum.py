"""Lessons 1.16-1.18 - PID control of a gravity-loaded joint (the pendulum).

The simulator is complete; the controller is yours.

Run:  python pid_pendulum.py                       # P → PD → PID comparison plot
      python pid_pendulum.py --gravity-comp        # same, with feedforward
      python pid_pendulum.py --disturb             # a shove at t=2.5s
"""
import sys
from pathlib import Path

import numpy as np

# Plant: m·l²·θ̈ = τ − m·g·l·sin(θ) − b·θ̇     (θ from downward vertical)
M, L, G, B = 0.5, 0.4, 9.81, 0.05
DT, T_END = 0.001, 5.0            # 1 kHz inner loop, 5 seconds
TARGET = np.pi / 2                 # hold horizontal; worst-case gravity load
TAU_MAX = 3.0                      # motor saturation (N·m); makes windup real


class PID:
    def __init__(self, kp, ki, kd, dt=DT, i_clamp=2.0):
        self.kp, self.ki, self.kd, self.dt, self.i_clamp = kp, ki, kd, dt, i_clamp
        self.integ, self.prev_err = 0.0, None

    def reset(self):
        self.integ, self.prev_err = 0.0, None

    def update(self, err):
        """Return the control output for this tick.

        TODO(you):
          1. P term: kp * err
          2. I term: accumulate err*dt into self.integ, CLAMP it to ±i_clamp
             (anti-windup), contribute ki * self.integ
          3. D term: kd * (err − prev_err)/dt   (use 0 on the first tick)
          4. remember prev_err; return the sum
        """
        raise NotImplementedError


def simulate(ctrl, gravity_comp=False, disturb=False):
    th, thd = 0.0, 0.0
    ts, ths = [], []
    ctrl.reset()
    for k in range(int(T_END / DT)):
        t = k * DT
        tau = ctrl.update(TARGET - th)
        if gravity_comp:
            tau += M * G * L * np.sin(th)          # feedforward the known load
        tau = np.clip(tau, -TAU_MAX, TAU_MAX)      # real motors saturate
        if disturb and abs(t - 2.5) < DT:
            thd += 2.0                             # the shove (rad/s, instantaneous)
        thdd = (tau - M * G * L * np.sin(th) - B * thd) / (M * L * L)
        thd += thdd * DT
        th += thd * DT                             # semi-implicit Euler
        ts.append(t); ths.append(th)
    return np.array(ts), np.array(ths)


def main():
    gravity_comp = "--gravity-comp" in sys.argv
    disturb = "--disturb" in sys.argv
    controllers = [
        ("P only",  PID(kp=8.0,  ki=0.0, kd=0.0), "#2a78d6"),
        ("PD",      PID(kp=8.0,  ki=0.0, kd=1.2), "#eb6834"),
        ("PID",     PID(kp=8.0,  ki=6.0, kd=1.2), "#1baf7a"),
    ]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for name, ctrl, color in controllers:
        ts, ths = simulate(ctrl, gravity_comp=gravity_comp, disturb=disturb)
        ax.plot(ts, np.degrees(ths), lw=2, color=color, label=name)
        final = np.degrees(np.mean(ths[-500:]))
        print(f"{name:8s} settles at {final:6.2f}°   (target {np.degrees(TARGET):.0f}°)")
    ax.axhline(np.degrees(TARGET), color="#52514e", lw=1, ls="--")
    ax.annotate("target", (T_END, np.degrees(TARGET)), textcoords="offset points",
                xytext=(-38, 6), color="#52514e")
    ax.set_xlabel("time (s)"); ax.set_ylabel("angle (deg)")
    suffix = " + gravity comp" if gravity_comp else ""
    ax.set_title(f"Holding a gravity-loaded joint{suffix}")
    ax.legend(frameon=False); ax.grid(color="#e8e7e3", lw=0.7)
    # Beside this script, not in the current directory: run from the repo root,
    # a bare filename would drop an untracked PNG there, and only
    # `module-*/code/*.png` is gitignored.
    out = Path(__file__).with_name("pid_run.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out.name} - compare the P droop, PD damping, PID convergence")


if __name__ == "__main__":
    main()
