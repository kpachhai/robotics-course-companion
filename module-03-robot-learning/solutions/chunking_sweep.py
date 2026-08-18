"""Lesson 3.6 - does predicting a chunk of actions actually help?

A 2D tracing task with second-order dynamics. An expert PD controller drives a
point mass round a circle. We clone it with an MLP that predicts k future
actions from one observation, execute all k open-loop, then re-query.

The interesting variable is `obs_noise`: how wrong the policy's view of the
world is. With a clean view, re-deciding every step wins outright. With a noisy
view, every query injects a fresh error, and querying less often wins - up to
the point where the policy has been blind for too long.

Rollouts are vectorised across episodes: `state` is (E, 4) and every episode
takes the same step at the same time, so a k=1 sweep costs one forward pass per
timestep instead of one per timestep per episode.

Run:  python chunking_sweep.py             the full sweep
      python chunking_sweep.py --quick     3 chunk sizes, 1 seed, for a check
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

DT = 0.05          # seconds per control step (20 Hz)
R = 1.0            # radius of the circle to trace
OMEGA = 1.0        # rad/s along the circle
KP, KD = 40.0, 9.0
AMAX = 12.0        # acceleration limit


# ---------------------------------------------------------------- the task

def expert(state):
    """PD controller: aim a little ahead on the circle, damp toward the tangent.

    A function of state only, so it is a legal thing for a Markov policy to
    imitate. It is also a genuinely good controller, which matters: the point
    of the experiment is that cloning a good controller is still hard.

    Works on a single (4,) state or a batch (E, 4).
    """
    state = np.atleast_2d(state)
    p, v = state[:, :2], state[:, 2:]
    theta = np.arctan2(p[:, 1], p[:, 0])
    ahead = theta + OMEGA * DT * 2
    p_target = R * np.stack([np.cos(ahead), np.sin(ahead)], axis=-1)
    v_target = R * OMEGA * np.stack([-np.sin(theta), np.cos(theta)], axis=-1)
    a = KP * (p_target - p) + KD * (v_target - v)
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return np.where(n > AMAX, a * AMAX / np.maximum(n, 1e-9), a)


def step(state, action):
    """Second-order point mass. Errors carry momentum; that is the whole point."""
    state = np.atleast_2d(state)
    p, v = state[:, :2], state[:, 2:]
    v = v + np.atleast_2d(action) * DT
    p = p + v * DT
    return np.concatenate([p, v], axis=-1)


def start_states(rng, n, jitter=0.0):
    theta = rng.uniform(0, 2 * np.pi, n)
    p = R * np.stack([np.cos(theta), np.sin(theta)], -1) + rng.normal(0, jitter, (n, 2))
    v = R * OMEGA * np.stack([-np.sin(theta), np.cos(theta)], -1) + rng.normal(0, jitter, (n, 2))
    return np.concatenate([p, v], -1)


def deviation(state):
    """How far off the circle, per episode."""
    return np.abs(np.linalg.norm(state[:, :2], axis=-1) - R)


# ------------------------------------------------------------ demonstrations

def make_dataset(k, obs_noise, n_demos=15, T=160, seed=0):
    """Observations paired with the next k expert actions, flattened.

    The expert acts on the true state; the log records a noisy view of it. That
    mismatch is the whole reason a cloned policy is worse than what it cloned.
    """
    rng = np.random.default_rng(seed)
    state = start_states(rng, n_demos, jitter=0.02)
    obs_log, act_log = [], []
    for _ in range(T):
        a = expert(state)
        obs_log.append(state + rng.normal(0, obs_noise, state.shape))
        act_log.append(a)
        state = step(state, a)
    O = np.stack(obs_log, 1)                       # (n_demos, T, 4)
    A = np.stack(act_log, 1)                       # (n_demos, T, 2)
    obs = np.concatenate([O[:, t] for t in range(T - k)])
    chunks = np.concatenate([A[:, t:t + k].reshape(n_demos, -1) for t in range(T - k)])
    return (torch.tensor(obs, dtype=torch.float32),
            torch.tensor(chunks, dtype=torch.float32))


# ------------------------------------------------------------------- policy

def train_policy(k, obs_noise, seed, epochs=40):
    """Clone the expert into an MLP that emits k actions at once."""
    torch.manual_seed(seed)
    X, Y = make_dataset(k, obs_noise, seed=seed)
    x_mean, x_std = X.mean(0), X.std(0) + 1e-6
    y_mean, y_std = Y.mean(0), Y.std(0) + 1e-6
    Xn, Yn = (X - x_mean) / x_std, (Y - y_mean) / y_std

    net = nn.Sequential(nn.Linear(4, 256), nn.ReLU(),
                        nn.Linear(256, 256), nn.ReLU(),
                        nn.Linear(256, 2 * k))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(epochs):
        perm = torch.randperm(len(Xn))
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            loss = ((net(Xn[b]) - Yn[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    def policy(observations):
        with torch.no_grad():
            x = (torch.tensor(observations, dtype=torch.float32) - x_mean) / x_std
            out = net(x) * y_std + y_mean
            return out.numpy().reshape(len(observations), k, 2)

    return policy


# ------------------------------------------------------------------ rollout

def rollout(policy, k, obs_noise, T=300, seed=99, n_episodes=25):
    """Closed loop, executing each chunk open-loop before re-querying.

    Returns an (n_episodes, T) array of distance from the circle per step.
    """
    rng = np.random.default_rng(seed)
    state = start_states(rng, n_episodes)
    devs, t = [], 0
    while t < T:
        chunk = policy(state + rng.normal(0, obs_noise, state.shape))
        for j in range(k):
            if t >= T:
                break
            state = step(state, chunk[:, j])
            devs.append(deviation(state))
            t += 1
    return np.stack(devs, axis=1)


def expert_baseline(obs_noise, T=300, seed=99, n_episodes=25):
    """The controller we cloned, driven from the same noisy observations."""
    rng = np.random.default_rng(seed)
    state = start_states(rng, n_episodes)
    devs = []
    for _ in range(T):
        state = step(state, expert(state + rng.normal(0, obs_noise, state.shape)))
        devs.append(deviation(state))
    return np.stack(devs, axis=1)


# -------------------------------------------------------------------- sweep

KS = (1, 2, 3, 5, 8, 12, 20, 35, 60)
NOISES = (0.0, 0.05, 0.12)
TRACE_KS = (1, 5, 20)


def sweep(ks=KS, noises=NOISES, seeds=(0, 1, 2), verbose=True):
    """Mean deviation from the circle for every (obs_noise, k, seed)."""
    results = {"ks": list(ks), "noises": list(noises), "seeds": list(seeds),
               "mean_dev": {}, "expert": {}, "traces": {}}
    for noise in noises:
        results["expert"][str(noise)] = float(expert_baseline(noise).mean())
        per_k = []
        for k in ks:
            per_seed = []
            for seed in seeds:
                t0 = time.time()
                policy = train_policy(k, noise, seed)
                D = rollout(policy, k, noise)
                per_seed.append(float(D.mean()))
                if seed == seeds[0] and k in TRACE_KS:
                    results["traces"][f"{noise}_{k}"] = D.mean(axis=0).tolist()
                if verbose:
                    print(f"  noise={noise:<5} k={k:<3} seed={seed}  "
                          f"mean dev {per_seed[-1]:.4f}   ({time.time()-t0:.0f}s)",
                          flush=True)
            per_k.append(per_seed)
        results["mean_dev"][str(noise)] = per_k
        if verbose:
            best = ks[int(np.argmin([np.mean(s) for s in per_k]))]
            print(f"  -> obs_noise={noise}: best chunk size k={best}\n", flush=True)
    return results


def summarise(res):
    lines = []
    for noise in res["noises"]:
        rows = np.array(res["mean_dev"][str(noise)]).mean(axis=1)
        best = res["ks"][int(np.argmin(rows))]
        lines.append(f"obs_noise={noise}: expert {res['expert'][str(noise)]:.4f} | "
                     f"k=1 {rows[0]:.4f} | best k={best} {rows.min():.4f} | "
                     f"k={res['ks'][-1]} {rows[-1]:.4f}")
    return "\n".join(lines)


RESULTS = Path(__file__).with_name("chunking_sweep_results.json")


def load_or_run(**kw):
    """Used by the figure generator so a rerun does not retrain everything."""
    if RESULTS.exists():
        return json.loads(RESULTS.read_text())
    res = sweep(**kw)
    RESULTS.write_text(json.dumps(res))
    return res


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    t0 = time.time()
    res = sweep(ks=TRACE_KS if quick else KS,
                noises=(0.0, 0.05) if quick else NOISES,
                seeds=(0,) if quick else (0, 1, 2))
    # A reduced run must never land on the committed measurement: the lesson
    # figures read that file, and a quiet downgrade to one seed would show up
    # as a mysteriously noisier plot rather than as an error.
    out = RESULTS.with_name("chunking_sweep_quick.json") if quick else RESULTS
    out.write_text(json.dumps(res))
    print(f"wrote {out}  ({time.time() - t0:.0f}s total)")
    print(summarise(res))
