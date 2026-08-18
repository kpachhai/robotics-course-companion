"""Lesson 3.6 - does predicting a chunk of actions actually help?

A 2D tracing task with second-order dynamics. An expert PD controller drives a
point mass round a circle. You clone it with an MLP that predicts k future
actions from one observation, execute all k open-loop, then re-query.

The task, the expert and the data are written for you. You write the two pieces
that decide the answer: the cloned policy, and the loop that executes a chunk.

Run:  python chunking_sweep.py --quick     3 chunk sizes, 1 seed
      python chunking_sweep.py             the full sweep
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
    imitate. Works on a single (4,) state or a batch (E, 4).
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
    """Clone the expert into an MLP that emits k actions at once.

    Returns a callable taking observations (E, 4) and returning (E, k, 2).
    """
    torch.manual_seed(seed)
    X, Y = make_dataset(k, obs_noise, seed=seed)

    # TODO(you): normalise X and Y (per-column mean and standard deviation).
    #            Skipping this is the single most common reason a behaviour
    #            cloning run looks broken: the loss is dominated by whichever
    #            column happens to have the largest units.

    # TODO(you): build an MLP 4 -> 256 -> 256 -> 2*k with ReLU, and train it
    #            with Adam(lr=1e-3) and mean-squared error for `epochs` passes
    #            in minibatches of 256.

    def policy(observations):
        # TODO(you): normalise, forward, un-normalise, reshape to (E, k, 2).
        raise NotImplementedError

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
        # TODO(you): query the policy ONCE with a noisy view of `state`, then
        #            execute all k returned actions with no further queries.
        #            Record deviation(state) after every step, and stop at T.
        raise NotImplementedError
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
    results = {"ks": list(ks), "noises": list(noises), "seeds": list(seeds),
               "mean_dev": {}, "expert": {}, "traces": {}}
    for noise in noises:
        results["expert"][str(noise)] = float(expert_baseline(noise).mean())
        per_k = []
        for k in ks:
            per_seed = []
            for seed in seeds:
                t0 = time.time()
                D = rollout(train_policy(k, noise, seed), k, noise)
                per_seed.append(float(D.mean()))
                if seed == seeds[0] and k in TRACE_KS:
                    results["traces"][f"{noise}_{k}"] = D.mean(axis=0).tolist()
                if verbose:
                    print(f"  noise={noise:<5} k={k:<3} seed={seed}  "
                          f"mean dev {per_seed[-1]:.4f}   ({time.time()-t0:.0f}s)",
                          flush=True)
            per_k.append(per_seed)
        results["mean_dev"][str(noise)] = per_k
    return results


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    res = sweep(ks=TRACE_KS if quick else KS,
                noises=(0.0, 0.05) if quick else NOISES,
                seeds=(0,) if quick else (0, 1, 2))
    name = "chunking_sweep_quick.json" if quick else "chunking_sweep_results.json"
    Path(__file__).with_name(name).write_text(json.dumps(res))
    for noise in res["noises"]:
        rows = np.array(res["mean_dev"][str(noise)]).mean(axis=1)
        best = res["ks"][int(np.argmin(rows))]
        print(f"obs_noise={noise}: expert {res['expert'][str(noise)]:.4f} | "
              f"k=1 {rows[0]:.4f} | best k={best} {rows.min():.4f}")
