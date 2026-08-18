"""Solution - Lesson 3.15: REINFORCE, a baseline, and PPO on one cart-pole.

Run:  python policy_gradient.py reinforce | baseline | ppo | noclip
      python policy_gradient.py variance | seeds | stability

Measured on a 2018 4-core laptop CPU under load; expect a little better idle.
"""
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from cartpole import MAX_T, VecCartPole

torch.set_num_threads(4)
GAMMA, CLIP_EPS, N_ENVS = 0.99, 0.2, 16
SOLVED = 195.0


def mlp(n_in, n_out, hidden=64):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, n_out))


def log_probs(policy, obs, act):
    return torch.log_softmax(policy(obs), -1).gather(1, act[:, None]).squeeze(1)


def collect(env, policy):
    obs = env.reset()
    O, A, R, M = [], [], [], []
    for _ in range(MAX_T):
        with torch.no_grad():
            logits = policy(torch.as_tensor(obs, dtype=torch.float32))
            act = torch.distributions.Categorical(logits=logits).sample().numpy()
        alive = env.alive.copy()
        O.append(obs); A.append(act); M.append(alive)
        obs, rew, _ = env.step(act)
        R.append(rew * alive)
        if not env.alive.any():
            break
    return (np.array(O, np.float32), np.array(A),
            np.array(R, np.float32), np.array(M))


def returns_to_go(rew, gamma=GAMMA):
    out = np.zeros_like(rew)
    running = np.zeros(rew.shape[1], np.float32)
    for t in reversed(range(len(rew))):
        running = rew[t] + gamma * running
        out[t] = running
    return out


def flatten(O, A, R, M):
    flat = M.reshape(-1)
    return (torch.as_tensor(O.reshape(-1, 4)[flat]),
            torch.as_tensor(A.reshape(-1)[flat]),
            torch.as_tensor(returns_to_go(R).reshape(-1)[flat]))


# --------------------------------------------------------------- the updates
def update_reinforce(policy, opt, obs, act, ret):
    loss = -(log_probs(policy, obs, act) * ret).mean()
    opt.zero_grad(); loss.backward(); opt.step()


def update_baseline(policy, value, opt, obs, act, ret):
    v = value(obs).squeeze(1)
    adv = (ret - v).detach()
    loss = -(log_probs(policy, obs, act) * adv).mean() + 0.5 * ((v - ret) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()


def update_ppo(policy, value, opt, obs, act, ret, rng,
               epochs=4, minibatch=512, clip=True):
    """Reuse the same batch `epochs` times. With clip=False this is the same
    reuse with no trust region, which is the ablation the lesson runs."""
    with torch.no_grad():
        old_logp = log_probs(policy, obs, act)
        adv = ret - value(obs).squeeze(1)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    idx = np.arange(len(obs))
    for _ in range(epochs):
        rng.shuffle(idx)
        for start in range(0, len(idx), minibatch):
            mb = torch.as_tensor(idx[start:start + minibatch])
            ratio = torch.exp(log_probs(policy, obs[mb], act[mb]) - old_logp[mb])
            gain = ratio * adv[mb]
            if clip:
                gain = torch.min(gain,
                                 torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv[mb])
            v_loss = ((value(obs[mb]).squeeze(1) - ret[mb]) ** 2).mean()
            opt.zero_grad(); (-gain.mean() + 0.5 * v_loss).backward(); opt.step()


# ---------------------------------------------------------------- the driver
def train(kind, seed=0, iters=80, lr=3e-3, verbose=True, stop_when_solved=True):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    env = VecCartPole(N_ENVS, rng)
    policy = mlp(4, 2)
    value = mlp(4, 1) if kind in ("baseline", "ppo", "noclip") else None
    params = list(policy.parameters()) + (list(value.parameters()) if value else [])
    opt = torch.optim.Adam(params, lr=lr)

    lengths, env_steps, curve, solved_at = [], 0, [], None
    started = time.perf_counter()
    for _ in range(iters):
        O, A, R, M = collect(env, policy)
        lengths += list(R.sum(0))
        env_steps += int(M.sum())
        obs, act, ret = flatten(O, A, R, M)

        if kind == "reinforce":
            update_reinforce(policy, opt, obs, act, ret)
        elif kind == "baseline":
            update_baseline(policy, value, opt, obs, act, ret)
        else:
            update_ppo(policy, value, opt, obs, act, ret, rng, clip=(kind == "ppo"))

        mean_len = float(np.mean(lengths[-N_ENVS * 3:]))
        curve.append((env_steps, mean_len))
        if solved_at is None and mean_len >= SOLVED:
            solved_at = env_steps
            if stop_when_solved:
                break                 # stop the clock the moment it is solved

    if verbose:
        solved = f"{solved_at:,}" if solved_at else "not solved"
        print(f"{kind:<10} seed={seed}  env steps used = {env_steps:,}  "
              f"solved at = {solved}  mean length = {curve[-1][1]:.0f}  "
              f"wall = {time.perf_counter() - started:.1f}s")
    return curve, solved_at


# ----------------------------------------- what actually kills the variance
def gradient_variance(seed=0, warmup=20, samples=32):
    """Freeze one policy, then measure how far three gradient estimators move
    from batch to batch.

    All three are unbiased estimates of the SAME gradient. Only their spread
    differs, which is the whole point. (Do not compare their sample means: the
    mean of a high-variance estimator is itself noisy at 32 samples, so it says
    nothing.)
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    env = VecCartPole(N_ENVS, rng)
    policy, value = mlp(4, 2), mlp(4, 1)
    opt = torch.optim.Adam(list(policy.parameters()) + list(value.parameters()), lr=3e-3)
    for _ in range(warmup):                       # get off the initialisation
        update_baseline(policy, value, opt, *flatten(*collect(env, policy)))

    def grad_vector(weights, obs, act):
        policy.zero_grad()
        (-(log_probs(policy, obs, act) * weights).mean()).backward()
        return torch.cat([p.grad.reshape(-1) for p in policy.parameters()]).clone()

    whole, to_go, minus_b = [], [], []
    for _ in range(samples):
        O, A, R, M = collect(env, policy)
        obs, act, ret = flatten(O, A, R, M)
        # the literal REINFORCE formula: every action in an episode carries that
        # episode's whole return, including reward banked before it was taken
        episode_return = np.ascontiguousarray(
            np.broadcast_to(R.sum(0), R.shape).reshape(-1)[M.reshape(-1)])
        with torch.no_grad():
            adv = ret - value(obs).squeeze(1)
        whole.append(grad_vector(torch.as_tensor(episode_return), obs, act))
        to_go.append(grad_vector(ret, obs, act))
        minus_b.append(grad_vector(adv, obs, act))

    def total_variance(vs):
        return float(torch.stack(vs).var(dim=0, unbiased=True).sum())

    v_whole, v_to_go, v_base = map(total_variance, (whole, to_go, minus_b))
    print(f"spread of the gradient estimate over {samples} independent batches, "
          f"one frozen policy")
    print(f"  weight = R(tau), the whole episode return   total variance = {v_whole:8.3f}")
    print(f"  weight = G_t, the return still to come      total variance = {v_to_go:8.3f}"
          f"   ({v_whole / v_to_go:.1f}x smaller)")
    print(f"  weight = G_t - V(s_t)                       total variance = {v_base:8.3f}"
          f"   ({v_to_go / v_base:.1f}x smaller again)")
    assert v_to_go < v_whole, "dropping the past should cut the variance"
    assert v_base < v_to_go, "a state baseline should cut it further"
    return v_whole, v_to_go, v_base


def seed_sweep(seeds=(0, 1, 2), kinds=("reinforce", "baseline", "ppo", "noclip"),
               iters=80, stop_when_solved=True):
    """Three seeds per algorithm - the minimum that lets you say anything, and
    still not enough to separate estimators this close together."""
    out = {}
    for kind in kinds:
        rows = [train(kind, seed=s, iters=iters, stop_when_solved=stop_when_solved)
                for s in seeds]
        out[kind] = rows
        got = [r[1] for r in rows if r[1]]
        median = f"{int(np.median(got)):,}" if got else "never"
        print(f"  == {kind}: solved at {got}, median {median}\n")
    return out


def stability(seeds=(0, 1, 2), iters=80):
    """Run past the finish line. Clipping is not about getting there faster;
    it is about not falling over once you have arrived."""
    for kind in ("ppo", "noclip"):
        for seed in seeds:
            curve, solved = train(kind, seed=seed, iters=iters, verbose=False,
                                  stop_when_solved=False)
            after = [m for steps, m in curve if solved and steps >= solved]
            worst = f"{min(after):.0f}" if after else "n/a"
            print(f"{kind:<8} seed={seed}  solved at {solved:,}  "
                  f"final {curve[-1][1]:.0f}  worst after solving {worst}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "reinforce"
    if which == "variance":
        gradient_variance()
    elif which == "seeds":
        seed_sweep()
    elif which == "stability":
        stability()
    else:
        train(which)
