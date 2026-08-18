"""Solution - Lesson 3.18: one task, three artifacts, one scoreboard.

Run:  python three_ways.py            the full scoreboard (about 100 s on a laptop CPU)
      python three_ways.py --quick    a smaller search budget, about 40 s
      python three_ways.py --warm     also search starting from the clone's weights

Everything here runs against `groove_world.py`, the 2-link arm dragging a pin
along a milled groove, so the three approaches face exactly the same task, the
same observation and the same action space. The only thing that differs is what
a human had to author.
"""
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

# groove_world.py ships complete under code/, so the solution reads it from there
# rather than keeping a second copy in step with it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import groove_world as gw  # noqa: E402

HORIZON = 150            # steps allowed per evaluation episode; the expert needs ~46
OFFSETS = (0.0, 0.02, 0.04)
N_EVAL = 60
SIZES = (2, 16, 2)       # the searched policy: 82 numbers
OUT_SCALE = 0.12         # cap on a commanded joint step; the expert peaks at 0.098
DEMOS = 50


# --------------------------------------------------------------- the predicate
def episode(policy, theta0, horizon=HORIZON):
    """Run one episode. Success = the pin reaches B without ever leaving the groove.

    The order matters. A policy that leaves the groove and then wanders back
    across B has not done the task, so arrival only counts if it happens before
    the first departure. Writing the predicate down like this, once, is what
    makes the three approaches comparable at all.
    """
    r = gw.rollout(policy, theta0, steps=horizon)
    n = len(r["pins"])
    if n == 0:
        return False, "diverged", 0
    left = np.flatnonzero(r["devs"] > gw.BAND)
    first_out = int(left[0]) if len(left) else n
    at_b = np.flatnonzero(np.linalg.norm(r["pins"] - gw.B_TIP, axis=1) < gw.REACH)
    arrived = int(at_b[0]) if len(at_b) else None
    if arrived is not None and arrived < first_out:
        return True, None, arrived
    return False, ("left_groove" if first_out < n else "never_arrived"), first_out


def sweep(policy, n=N_EVAL, offsets=OFFSETS, seed=7):
    """Success rate at each starting offset, plus the failure histogram."""
    out = {}
    for off in offsets:
        rng = np.random.default_rng(seed)
        runs = [episode(policy, gw.start_pose(rng, off)) for _ in range(n)]
        out[off] = dict(k=sum(r[0] for r in runs), n=n,
                        fails=Counter(r[1] for r in runs if not r[0]))
    return out


# ------------------------------------------------------- artifact 1: the rule
def scripted():
    """Nothing to train. `gw.expert` IS the artifact: pure pursuit along a known path."""
    return gw.expert, dict(demos=0, transitions=0, env_steps=0, seconds=0.0)


# ---------------------------------------------- artifact 2: the demonstrations
def imitate(n_demos=DEMOS, hidden=128, epochs=100, seed=0):
    """Behaviour cloning, straight out of lesson 3.3, on `n_demos` episodes."""
    t0 = time.perf_counter()
    states, actions = gw.collect(n_demos, seed=seed)
    policy, history = gw.fit_policy(states, actions, epochs=epochs, seed=seed, hidden=hidden)
    return policy, dict(demos=n_demos, transitions=len(states), env_steps=0,
                        seconds=time.perf_counter() - t0, val=float(history[-1][1]))


# ---------------------------------------------------- artifact 3: the reward
def unpack(vec, sizes=SIZES):
    layers, i = [], 0
    for a, b in zip(sizes[:-1], sizes[1:]):
        W = vec[i:i + a * b].reshape(a, b); i += a * b
        c = vec[i:i + b]; i += b
        layers.append((W, c))
    return layers


def n_params(sizes=SIZES):
    return sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))


def make_net(vec, sizes=SIZES, out_scale=OUT_SCALE):
    """Turn a flat parameter vector into a policy with `gw.expert`'s signature."""
    layers = unpack(vec, sizes)

    def policy(theta):
        h = (np.asarray(theta, dtype=float) - gw.HOME) / 0.6
        for k, (W, b) in enumerate(layers):
            h = h @ W + b
            if k < len(layers) - 1:
                h = np.tanh(h)
        return np.tanh(h) * out_scale
    return policy


def path_progress(pins):
    """How far along the groove a run got, as a fraction. Vectorised on purpose."""
    d = ((gw.PATH[None, :, :] - pins[:, None, :]) ** 2).sum(-1)
    return float(d.argmin(1).max() / (len(gw.PATH) - 1))


def reward(vec, starts, horizon=HORIZON):
    """Groove covered BEFORE the pin first left it, plus 1 for finishing.

    The "before" is not decoration. Two earlier versions of this function were
    both gamed inside a minute of search: one measured the furthest path point
    ever approached, so the policy cut the corner straight to B; the next
    measured progress over in-band samples only, so the policy left the groove,
    flew to B, and landed back in band at the far end. Lesson 3.14 is about
    exactly this, and writing a reward for a task you can simply demonstrate is
    how you end up needing it.
    """
    policy = make_net(vec)
    total, used = 0.0, 0
    for theta0 in starts:
        r = gw.rollout(policy, theta0, steps=horizon)
        n = len(r["pins"])
        used += max(n, 1)
        if n == 0:
            continue
        left = np.flatnonzero(r["devs"] > gw.BAND)
        cut = int(left[0]) if len(left) else n
        if cut == 0:
            continue
        at_b = np.flatnonzero(np.linalg.norm(r["pins"][:cut] - gw.B_TIP, axis=1) < gw.REACH)
        total += path_progress(r["pins"][:cut]) + (1.0 if len(at_b) else 0.0)
    return total / len(starts), used


def elite_update(pops, scores, elite):
    """Refit the sampling distribution to the best `elite` candidates."""
    order = np.argsort(scores)[::-1][:elite]
    winners = pops[order]
    mu = winners.mean(0)
    sigma = winners.std(0) + 0.02      # the floor is what keeps it exploring
    return order, mu, sigma


def search(budget=200_000, pop=48, elite=8, sigma0=None, n_train=4, seed=0,
           start=None, verbose=True):
    """Cross-entropy method: sample policies, keep the best few, resample around them.

    No demonstrations anywhere. The only thing this reads is the reward, and the
    only currency it spends is environment steps, which is what `budget` counts.
    """
    rng = np.random.default_rng(seed)
    starts = [gw.start_pose(np.random.default_rng(1000 + k)) for k in range(n_train)]
    d = n_params()
    mu = np.zeros(d) if start is None else np.asarray(start, dtype=float)
    # From noise, spread wide. From a policy that already works, do not: the first
    # generation would sample 48 corruptions of it and throw the head start away.
    sigma = np.full(d, sigma0 if sigma0 is not None else (0.4 if start is None else 0.05))
    used, gen, best, curve = 0, 0, (-1e9, mu.copy()), []
    while used < budget:
        pops = rng.normal(mu, sigma, (pop, d))
        scores = np.zeros(pop)
        for j, vec in enumerate(pops):
            scores[j], spent = reward(vec, starts)
            used += spent
        order, mu, sigma = elite_update(pops, scores, elite)
        if scores[order[0]] > best[0]:
            best = (float(scores[order[0]]), pops[order[0]].copy())
        gen += 1
        curve.append((used, float(scores[order].mean()), best[0]))
        if verbose and gen % 20 == 0:
            print(f"    generation {gen:>3}  {used:>9,} env steps  "
                  f"elite {scores[order].mean():.3f}  best {best[0]:.3f}")
    return best[1], used, curve


def reinforce(budget=200_000, seed=0, start=None, verbose=True):
    t0 = time.perf_counter()
    vec, used, curve = search(budget=budget, seed=seed, start=start, verbose=verbose)
    return make_net(vec), dict(demos=0, transitions=0, env_steps=used,
                               seconds=time.perf_counter() - t0,
                               best_reward=curve[-1][2], generations=len(curve),
                               curve=curve, vector=vec)


# ------------------------------------- the same small policy, cloned not searched
def clone_small(n_demos=DEMOS, epochs=400, seed=0):
    """Fit the SEARCHED architecture by imitation, so the two are comparable.

    Returns the policy and its weights flattened the way `make_net` unpacks them,
    which is what lets the search start from the clone instead of from noise.
    """
    states, actions = gw.collect(n_demos, seed=seed)
    torch.manual_seed(seed)
    X = torch.tensor((states - gw.HOME) / 0.6, dtype=torch.float32)
    Y = torch.tensor(actions, dtype=torch.float32)
    net = nn.Sequential(nn.Linear(2, SIZES[1]), nn.Tanh(), nn.Linear(SIZES[1], 2), nn.Tanh())
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    for _ in range(epochs):
        loss = ((net(X) * OUT_SCALE - Y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    flat = np.concatenate([(p.detach().numpy().T if p.ndim == 2 else p.detach().numpy()).ravel()
                           for p in net.parameters()])
    return make_net(flat), flat, float(loss.item())


# ------------------------------------------------------------------ reporting
def bench(budget=200_000, verbose=True):
    """Run all three, once. The printed table and the lesson's figures share this."""
    torch.set_num_threads(4)
    rows = []

    rule, cost = scripted()
    rows.append(dict(tag="scripted", authored="a path and a tracker",
                     consumed="nothing", cost=cost, result=sweep(rule)))

    for n_demos in (10, DEMOS):
        clone, cost = imitate(n_demos=n_demos)
        rows.append(dict(tag=f"imitation, {n_demos}", authored=f"{n_demos} demonstrations",
                         consumed=f"{cost['transitions']:,} transitions, "
                                  f"{cost['seconds']:.0f} s",
                         cost=cost, result=sweep(clone)))

    searched, cost = reinforce(budget=budget, verbose=verbose)
    rows.append(dict(tag="reinforcement", authored="a reward function",
                     consumed=f"{cost['env_steps']:,} env steps, {cost['seconds']:.0f} s",
                     cost=cost, result=sweep(searched)))
    return rows


def line(row):
    cells = "   ".join(f"{row['result'][o]['k']:>2}/{row['result'][o]['n']:<3}"
                       for o in OFFSETS)
    print(f"  {row['tag']:<18}{row['authored']:<30}{row['consumed']:<30}{cells}")


def main(budget=200_000, warm=False):
    print(f"\ngroove world: success = pin reaches B without leaving the 5 cm band, "
          f"{HORIZON}-step horizon, {N_EVAL} trials per cell")
    print(f"\n  {'approach':<18}{'you authored':<30}{'it consumed':<30}"
          f"on centre    +2 cm    +4 cm")

    rows = bench(budget=budget)
    for row in rows:
        line(row)
    last = rows[-1]["cost"]
    print(f"    best reward reached {last['best_reward']:.3f} of a possible 2.000 "
          f"over {last['generations']} generations")

    if warm:
        small, flat, mse = clone_small()
        line(dict(tag="clone, 82 params", authored=f"{DEMOS} demonstrations",
                  consumed=f"training MSE {mse:.2e}", result=sweep(small)))
        polished, cost = reinforce(budget=budget, start=flat, verbose=False)
        line(dict(tag="clone, then search", authored="demonstrations, then a reward",
                  consumed=f"+{cost['env_steps']:,} env steps", result=sweep(polished)))


if __name__ == "__main__":
    main(budget=60_000 if "--quick" in sys.argv else 200_000, warm="--warm" in sys.argv)
