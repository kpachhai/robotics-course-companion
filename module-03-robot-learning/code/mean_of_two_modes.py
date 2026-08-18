"""Lesson 3.9 - what a squared-error loss actually learns, and when that is fatal.

Three experiments on the 2-link arm from Module 1, all on a laptop CPU:

  1. forward     theta -> xy      ONE right answer per input     fits
  2. inverse     xy -> theta      TWO right answers per input    stalls, and the
                                  pose it predicts reaches nothing
  3. inverse, one branch only     ONE right answer per input     fits

Experiment 3 is the control. It rules out "the network is too small" and "the
optimiser is broken", because it is the same network, the same optimiser and the
same number of steps on the same map - with one of the two branches deleted.

Angles wrap at +-pi, and a squared-error loss does not know that: -179 and +179
degrees are two degrees apart and the loss thinks they are 358. So the networks
here predict cos and sin of each joint angle and we recover the angle with
atan2. That removes wrapping as a confound, leaving only the effect we came for.

Starter. Three pieces are left for you, marked TODO(you). The first one has to
be filled in before anything runs at all.

Run:  python mean_of_two_modes.py
      python mean_of_two_modes.py --sweep    (adds the branch-mixture sweep)
"""
import sys
import time

import numpy as np
import torch
import torch.nn as nn

L1, L2 = 1.0, 0.7
SEED = 0


# --------------------------------------------------------------------------
# the arm

def fk(theta):
    """theta: (N, 2) joint angles -> (N, 2) end-effector xy."""
    t1, t2 = theta[:, 0], theta[:, 1]
    return np.stack([L1 * np.cos(t1) + L2 * np.cos(t1 + t2),
                     L1 * np.sin(t1) + L2 * np.sin(t1 + t2)], axis=1)


def sample_arm(n, rng, elbow=None):
    """Random reachable (theta, xy) pairs.

    elbow=None  both branches, so xy -> theta is two-to-one
    elbow=+1    theta2 > 0 only, one branch, so xy -> theta is one-to-one
    """
    t1 = rng.uniform(-np.pi, np.pi, n)
    t2 = rng.uniform(0.35, np.pi - 0.35, n)          # away from both singularities
    # TODO(you): elbow=None should flip the sign of t2 at random, so both
    #            branches appear. elbow=+1 or -1 should force one sign, which
    #            is the control that makes this an experiment.
    raise NotImplementedError


def as_circle(theta):
    """(N, 2) angles -> (N, 4) of [cos, cos, sin, sin]. No wrap discontinuity."""
    return np.concatenate([np.cos(theta), np.sin(theta)], axis=1)


def from_circle(z):
    """(N, 4) -> (N, 2) angles."""
    return np.arctan2(z[:, 2:], z[:, :2])


# --------------------------------------------------------------------------
# the smallest useful regressor

def mlp(d_in, d_out, width=256):
    return nn.Sequential(nn.Linear(d_in, width), nn.ReLU(),
                         nn.Linear(width, width), nn.ReLU(),
                         nn.Linear(width, d_out))


def fit(x, y, steps=2000, width=256, batch=128, lr=2e-3, seed=SEED):
    """Plain supervised regression under mean squared error."""
    torch.manual_seed(seed)
    net = mlp(x.shape[1], y.shape[1], width)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    xt = torch.as_tensor(x, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)

    losses = []
    for _ in range(steps):
        idx = torch.randint(0, len(xt), (batch,), generator=g)
        loss = ((net(xt[idx]) - yt[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return net, np.array(losses)


@torch.no_grad()
def predict(net, x):
    return net(torch.as_tensor(x, dtype=torch.float32)).numpy()


# --------------------------------------------------------------------------
# experiments

def run(steps=2000):
    torch.set_num_threads(2)
    theta, xy = sample_arm(20000, np.random.default_rng(SEED))          # both branches
    theta_v, xy_v = sample_arm(4000, np.random.default_rng(SEED + 1))
    theta_u, xy_u = sample_arm(20000, np.random.default_rng(SEED + 2), elbow=+1)
    theta_uv, xy_uv = sample_arm(4000, np.random.default_rng(SEED + 3), elbow=+1)

    out = {}

    t0 = time.time()
    net_f, loss_f = fit(theta, xy, steps=steps)
    err_f = predict(net_f, theta_v) - xy_v
    out["forward"] = dict(mse=float((err_f ** 2).mean()), curve=loss_f,
                          secs=time.time() - t0,
                          miss=np.linalg.norm(err_f, axis=1))

    for name, (xa, ya, xb, yb) in {
        "inverse": (xy, theta, xy_v, theta_v),
        "inverse_one_branch": (xy_u, theta_u, xy_uv, theta_uv),
    }.items():
        t0 = time.time()
        net, curve = fit(xa, as_circle(ya), steps=steps)
        pred = predict(net, xb)
        # The damning check: put the PREDICTED joint angles back through forward
        # kinematics and see where the arm actually ends up.
        reached = fk(from_circle(pred))
        out[name] = dict(mse=float(((pred - as_circle(yb)) ** 2).mean()), curve=curve,
                         secs=time.time() - t0,
                         miss=np.linalg.norm(reached - xb, axis=1),
                         pred_theta=from_circle(pred))
    return out


def report(out):
    print(f"{'experiment':<22}{'val MSE':>10}{'median miss':>14}{'worst 10%':>12}{'train s':>10}")
    for key in ("forward", "inverse", "inverse_one_branch"):
        r = out[key]
        print(f"{key:<22}{r['mse']:>10.4f}{np.median(r['miss']):>13.3f} m"
              f"{np.quantile(r['miss'], 0.9):>11.3f} m{r['secs']:>10.1f}")
    print(f"\nThe arm is {L1 + L2:.1f} m long. A "
          f"{np.median(out['inverse']['miss']):.2f} m miss is "
          f"{100 * np.median(out['inverse']['miss']) / (L1 + L2):.0f}% of its reach.")
    print("'miss' is where the end-effector lands when you drive the arm to the "
          "predicted angles.")


# --------------------------------------------------------------------------
# the two exact solutions for one target, and their average

def two_solutions(x, y):
    r2 = x * x + y * y
    c2 = (r2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    t2 = np.arccos(np.clip(c2, -1, 1))
    out = []
    for s in (+1.0, -1.0):
        th2 = s * t2
        th1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(th2), L1 + L2 * np.cos(th2))
        out.append((th1, th2))
    return out


def show_average(x=0.9, y=0.6):
    (a1, a2), (b1, b2) = two_solutions(x, y)
    mean = np.array([[(a1 + b1) / 2, (a2 + b2) / 2]])
    print(f"\ntarget ({x}, {y}) has exactly two solutions:")
    # positive theta2 puts the elbow BELOW the line from base to target
    for name, t in (("elbow down", (a1, a2)), ("elbow up", (b1, b2))):
        p = fk(np.array([t]))[0]
        print(f"  {name:<11} theta = ({t[0]:+.3f}, {t[1]:+.3f})  reaches "
              f"({p[0]:+.3f}, {p[1]:+.3f})")
    p = fk(mean)[0]
    print(f"  {'their mean':<11} theta = ({mean[0, 0]:+.3f}, {mean[0, 1]:+.3f})  reaches "
          f"({p[0]:+.3f}, {p[1]:+.3f})   misses by {np.hypot(p[0] - x, p[1] - y):.3f} m")


# --------------------------------------------------------------------------
# finding the modes without being told they are there

def mode_histogram(target, theta, xy, radius=0.03, bins=36):
    """Every training sample whose target is within `radius` of this point.

    Histogram their second joint angle. Two spikes with a gap between them is
    what multimodality looks like when nobody has told you it is there. This is
    the diagnostic you run on real robot data.
    """
    # TODO(you): select the samples whose xy is within `radius` of `target`,
    #            histogram their theta2 over (-pi, pi), and return
    #            (counts, edges, how_many_samples).
    raise NotImplementedError


def show_modes(target=(0.9, 0.6)):
    theta, xy = sample_arm(60000, np.random.default_rng(SEED + 11))
    counts, edges, n = mode_histogram(target, theta, xy)
    print(f"\n{n} training samples land within 3 cm of {target}. "
          f"Their second joint angle:")
    peak = counts.max()
    for count, lo in zip(counts, edges):
        if count or abs(lo) < 0.4:
            bar = "#" * int(round(28 * count / max(peak, 1)))
            print(f"  {lo:+5.2f} rad {bar}")
    print("  two spikes, and nothing at zero. That gap is where the mean lands.")


def mixture_sweep(fractions=(1.0, 0.9, 0.75, 0.5), steps=800, n=20000):
    """How unbalanced do the two branches have to be before the fit recovers?"""
    # TODO(you): for each fraction, build a training set that is that fraction
    #            elbow-up and the rest elbow-down, fit, and measure the median
    #            miss distance against an elbow-up validation set. Print a row
    #            per fraction. Where does it stop being survivable?
    raise NotImplementedError


if __name__ == "__main__":
    out = run()
    report(out)
    show_average()
    show_modes()
    if "--sweep" in sys.argv:
        print("\nbranch mixture sweep (short runs, so the numbers are coarse):")
        mixture_sweep()
