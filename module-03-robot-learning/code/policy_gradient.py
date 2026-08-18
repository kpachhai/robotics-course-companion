"""Lesson 3.15 - REINFORCE, then a baseline, then PPO. Three update rules, one
task, one data budget each.

The environment, the rollout collection, the return-to-go computation and the
bookkeeping are done. What is yours is the loss each algorithm minimises -
which is the only thing that differs between them.

Run:  python policy_gradient.py reinforce
      python policy_gradient.py baseline
      python policy_gradient.py ppo
      python policy_gradient.py noclip          # PPO's reuse, without the clip
      python policy_gradient.py variance        # what actually kills the variance

One algorithm at a time. Each takes roughly 20-60 s on a laptop CPU.
"""
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from cartpole import MAX_T, VecCartPole

torch.set_num_threads(4)
GAMMA, CLIP_EPS, N_ENVS = 0.99, 0.2, 16
SOLVED = 195.0                     # mean episode length over the last three batches


def mlp(n_in, n_out, hidden=64):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, n_out))


def log_probs(policy, obs, act):
    """log pi(a_t | s_t) for the actions actually taken."""
    return torch.log_softmax(policy(obs), -1).gather(1, act[:, None]).squeeze(1)


def collect(env, policy):
    """One batch: N_ENVS complete episodes. Returns (obs, act, rew, mask) with
    time on axis 0 and environment on axis 1."""
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
    """G_t = sum_{k>=t} gamma^(k-t) r_k, computed backwards in one pass."""
    out = np.zeros_like(rew)
    running = np.zeros(rew.shape[1], np.float32)
    for t in reversed(range(len(rew))):
        running = rew[t] + gamma * running
        out[t] = running
    return out


# --------------------------------------------------------------- the updates
def update_reinforce(policy, opt, obs, act, ret):
    """Vanilla policy gradient.

    TODO(you):
      1. logp = log_probs(policy, obs, act)
      2. loss = -(logp * ret).mean()
         The minus sign is because torch minimises and we want to climb.
         `ret` is a constant here - never call .backward() through it.
      3. zero the optimiser, backward, step
    """
    raise NotImplementedError


def update_baseline(policy, value, opt, obs, act, ret):
    """Same gradient, graded on a curve.

    TODO(you):
      1. v = value(obs).squeeze(1)
      2. adv = (ret - v).detach()      # detach: the baseline is not a target
      3. policy loss = -(logp * adv).mean()
      4. value loss  = 0.5 * ((v - ret) ** 2).mean()
      5. one optimiser step on the sum
    """
    raise NotImplementedError


def update_ppo(policy, value, opt, obs, act, ret, rng,
               epochs=4, minibatch=512, clip=True):
    """Reuse the same batch several times without letting the policy run away.

    With clip=False this becomes the same reuse with no trust region, which is
    the ablation that shows what clipping actually buys.

    TODO(you):
      1. Under torch.no_grad(), record old_logp = log_probs(...) and
         v = value(obs).squeeze(1). These freeze the policy that collected
         the data.
      2. adv = ret - v, then normalise it: (adv - adv.mean()) / (adv.std() + 1e-8)
      3. For `epochs` passes over a shuffled index, in slices of `minibatch`:
           ratio = exp(logp_now - old_logp)
           gain  = ratio * adv
           if clip: gain = min(gain, clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * adv)
           policy loss = -gain.mean()
           value loss  = ((value(obs_mb).squeeze(1) - ret_mb) ** 2).mean()
           one optimiser step on policy loss + 0.5 * value loss
    """
    raise NotImplementedError


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
        flat = M.reshape(-1)
        obs = torch.as_tensor(O.reshape(-1, 4)[flat])
        act = torch.as_tensor(A.reshape(-1)[flat])
        ret = torch.as_tensor(returns_to_go(R).reshape(-1)[flat])

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
    from batch to batch. All three are unbiased estimates of the same gradient;
    only their spread differs.

    The setup is done. Three lines are yours, marked below.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    env = VecCartPole(N_ENVS, rng)
    policy, value = mlp(4, 2), mlp(4, 1)
    opt = torch.optim.Adam(list(policy.parameters()) + list(value.parameters()), lr=3e-3)
    for _ in range(warmup):                       # get off the initialisation
        O, A, R, M = collect(env, policy)
        flat = M.reshape(-1)
        update_baseline(policy, value, opt,
                        torch.as_tensor(O.reshape(-1, 4)[flat]),
                        torch.as_tensor(A.reshape(-1)[flat]),
                        torch.as_tensor(returns_to_go(R).reshape(-1)[flat]))

    def grad_vector(weights, obs, act):
        """The policy gradient with each log-prob weighted by `weights`."""
        policy.zero_grad()
        (-(log_probs(policy, obs, act) * weights).mean()).backward()
        return torch.cat([p.grad.reshape(-1) for p in policy.parameters()]).clone()

    whole, to_go, minus_b = [], [], []
    for _ in range(samples):
        O, A, R, M = collect(env, policy)
        flat = M.reshape(-1)
        obs = torch.as_tensor(O.reshape(-1, 4)[flat])
        act = torch.as_tensor(A.reshape(-1)[flat])
        ret = torch.as_tensor(returns_to_go(R).reshape(-1)[flat])
        # the literal REINFORCE formula: every action carries the whole episode
        # return, including the reward banked before that action was taken
        episode_return = torch.as_tensor(np.ascontiguousarray(
            np.broadcast_to(R.sum(0), R.shape).reshape(-1)[flat]))
        with torch.no_grad():
            adv = ret - value(obs).squeeze(1)
        # TODO(you): three calls to grad_vector, one per weighting, appended to
        # `whole`, `to_go` and `minus_b` respectively.
        raise NotImplementedError

    def total_variance(vs):
        """Sum over coordinates of the per-coordinate variance across batches."""
        return float(torch.stack(vs).var(dim=0, unbiased=True).sum())

    v_whole, v_to_go, v_base = map(total_variance, (whole, to_go, minus_b))
    print(f"spread of the gradient estimate over {samples} independent batches, "
          f"one frozen policy")
    print(f"  weight = R(tau), the whole episode return   total variance = {v_whole:8.3f}")
    print(f"  weight = G_t, the return still to come      total variance = {v_to_go:8.3f}"
          f"   ({v_whole / v_to_go:.1f}x smaller)")
    print(f"  weight = G_t - V(s_t)                       total variance = {v_base:8.3f}"
          f"   ({v_to_go / v_base:.1f}x smaller again)")
    return v_whole, v_to_go, v_base


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "reinforce"
    if which == "variance":
        gradient_variance()
    else:
        train(which)
