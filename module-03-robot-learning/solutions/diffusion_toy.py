"""Lesson 3.10 - a conditional diffusion model, small enough to read in one sitting.

The task is the corridor from lesson 3.9. One observation: how far ahead the
obstacle sits. One action: the sideways-and-forward step to take next. Half the
demonstrations go left, half go right, so the correct answer is a distribution
with two lumps and nothing in between.

  regressor   obs -> action            trained with mean squared error
  diffusion   obs -> p(action | obs)   trained with mean squared error TOO,
                                       on the noise rather than on the action

Everything here is CPU-sized. Run:

    python diffusion_toy.py            train both, print the numbers
    python diffusion_toy.py --sweep    add the inference-step sweep
"""
import sys
import time

import numpy as np
import torch
import torch.nn as nn

SEED = 0
K = 100                    # denoising steps, matching LeRobot's num_train_timesteps


# --------------------------------------------------------------------------
# the demonstrations

def actions_for(obs, rng):
    """The action a human demonstrated, given how far ahead the obstacle sits.

    Every demonstrator goes round the obstacle. Which side is a coin flip, and
    nothing in the observation says which - that information was in the
    demonstrator's head and was never written down.
    """
    side = rng.choice([-1.0, 1.0], (len(obs), 1))     # the coin flip, never observed
    sideways = side * (0.55 - 0.35 * obs)             # closer obstacle, bigger swerve
    forward = 0.25 + 0.10 * obs
    action = np.concatenate([sideways, forward], axis=1)
    action += rng.normal(0, 0.03, action.shape)       # human jitter
    return action.astype(np.float32)


def demos(n, rng):
    obs = rng.uniform(0.2, 1.0, (n, 1)).astype(np.float32)
    return obs, actions_for(obs, rng)


# --------------------------------------------------------------------------
# model 1: regress the action

def mlp(d_in, d_out, width=128, depth=3):
    layers, d = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.SiLU()]
        d = width
    return nn.Sequential(*layers, nn.Linear(width, d_out))


def train_regressor(obs, act, steps=2000, batch=256, lr=2e-3, seed=SEED):
    torch.manual_seed(seed)
    net = mlp(1, 2)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    o, a = torch.as_tensor(obs), torch.as_tensor(act)
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for _ in range(steps):
        i = torch.randint(0, len(o), (batch,), generator=g)
        loss = ((net(o[i]) - a[i]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net, time.time() - t0


# --------------------------------------------------------------------------
# model 2: diffuse the action
#
# Forward process (no learning, just arithmetic): take a real action and mix in
# noise, more of it at every step k, until nothing of the action is left.
#   a_k = sqrt(abar_k) * a_0  +  sqrt(1 - abar_k) * eps
# Reverse process (this is the network): given the noisy action, the step index
# and the observation, guess which noise was added. Then subtract a bit of it.

def schedule(k_steps=K, beta_lo=1e-4, beta_hi=0.02):
    betas = torch.linspace(beta_lo, beta_hi, k_steps)
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, dim=0)
    return betas, alphas, abar


def q_sample(a0, k, eps):
    """The forward process: spoil a real action by the amount step k calls for.

    Two checks worth running: at k=0 this returns a0 almost exactly, and at
    k=K-1 the result has unit standard deviation whatever a0 was.
    """
    _, _, abar = schedule()
    return abar[k].sqrt().unsqueeze(-1) * a0 + (1 - abar[k]).sqrt().unsqueeze(-1) * eps


class EpsNet(nn.Module):
    """Predicts the noise. Input: noisy action, step index, observation."""

    def __init__(self, d_act=2, d_obs=1, width=128):
        super().__init__()
        self.net = mlp(d_act + 2 + d_obs, d_act, width=width)

    def forward(self, a_k, k, obs):
        # Two numbers for the step index instead of one: a raw integer is a poor
        # input, and sin/cos of it gives the network a smooth handle on "how
        # noisy is this".
        t = k.float().unsqueeze(-1) / K
        emb = torch.cat([torch.sin(2 * np.pi * t), torch.cos(2 * np.pi * t)], dim=-1)
        return self.net(torch.cat([a_k, emb, obs], dim=-1))


def train_diffusion(obs, act, steps=2000, batch=256, lr=2e-3, seed=SEED):
    torch.manual_seed(seed)
    net = EpsNet()
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    _, _, abar = schedule()
    o, a = torch.as_tensor(obs), torch.as_tensor(act)
    g = torch.Generator().manual_seed(seed)

    t0 = time.time()
    for _ in range(steps):
        i = torch.randint(0, len(o), (batch,), generator=g)
        a0, ob = a[i], o[i]
        k = torch.randint(0, K, (batch,), generator=g)
        eps = torch.randn(a0.shape, generator=g)
        ak = q_sample(a0, k, eps)
        loss = ((net(ak, k, ob) - eps) ** 2).mean()      # still mean squared error
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net, time.time() - t0


@torch.no_grad()
def sample(net, obs, seed=SEED, keep=(), steps=K):
    """Start from pure noise and denoise. Returns (actions, snapshots).

    `keep` is a set of step indices to snapshot, for the figure that shows the
    cloud splitting in two.

    With steps=K this is the ancestral sampler from the lesson. With fewer, it
    walks a subsequence using the deterministic update instead, because the
    ancestral noise term is only calibrated for single steps.
    """
    betas, alphas, abar = schedule()
    g = torch.Generator().manual_seed(seed)
    ob = torch.as_tensor(obs)
    a = torch.randn((len(ob), 2), generator=g)
    ks = list(range(0, K, max(1, K // steps)))
    snaps = {}
    for i, k in reversed(list(enumerate(ks))):
        if k in keep:
            snaps[k] = a.clone().numpy()
        kk = torch.full((len(ob),), k, dtype=torch.long)
        eps = net(a, kk, ob)
        if len(ks) == K:
            a = (a - betas[k] / (1 - abar[k]).sqrt() * eps) / alphas[k].sqrt()
            if k > 0:
                a = a + betas[k].sqrt() * torch.randn(a.shape, generator=g)
        else:
            a0 = (a - (1 - abar[k]).sqrt() * eps) / abar[k].sqrt()
            prev = abar[ks[i - 1]] if i else torch.tensor(1.0)
            a = prev.sqrt() * a0 + (1 - prev).sqrt() * eps
    snaps[0] = a.numpy()
    return a.numpy(), snaps


def step_sweep(net, obs_at=0.5, n=1000, counts=(100, 50, 20, 10, 5)):
    """Latency against sample quality. The number you deploy with lives here."""
    probe = np.full((n, 1), obs_at, dtype=np.float32)
    print(f"\n{'steps':>7}{'left':>8}{'straight':>10}{'right':>8}{'us/action':>12}")
    for s in counts:
        t0 = time.time()
        acts, _ = sample(net, probe, steps=s)
        per = 1e6 * (time.time() - t0) / n
        left, mid, right = side_fractions(acts)
        print(f"{s:>7}{left:>8.0%}{mid:>10.0%}{right:>8.0%}{per:>12.1f}")


# --------------------------------------------------------------------------

def side_fractions(actions, dead=0.12):
    """What fraction of predicted actions go left, right, or straight ahead.

    'Straight ahead' is the failure: no demonstrator ever went there.
    """
    x = actions[:, 0]
    return (float((x < -dead).mean()), float((np.abs(x) <= dead).mean()),
            float((x > dead).mean()))


def run(steps=2000, n_eval=2000, obs_at=0.5):
    torch.set_num_threads(2)
    rng = np.random.default_rng(SEED)
    obs, act = demos(20000, rng)

    reg, t_reg = train_regressor(obs, act, steps=steps)
    dif, t_dif = train_diffusion(obs, act, steps=steps)

    probe = np.full((n_eval, 1), obs_at, dtype=np.float32)
    with torch.no_grad():
        t0 = time.time()
        reg_a = reg(torch.as_tensor(probe)).numpy()
        t_reg_inf = (time.time() - t0) / n_eval
    t0 = time.time()
    dif_a, snaps = sample(dif, probe, keep=(80, 60, 40, 20))
    t_dif_inf = (time.time() - t0) / n_eval

    truth = actions_for(probe, np.random.default_rng(SEED + 9))

    return dict(reg=reg_a, dif=dif_a, truth=truth, snaps=snaps, obs_at=obs_at,
                t_reg=t_reg, t_dif=t_dif, t_reg_inf=t_reg_inf, t_dif_inf=t_dif_inf,
                reg_net=reg, dif_net=dif, demos=(obs, act))


def report(r):
    print(f"training, 2000 steps each:  regressor {r['t_reg']:.1f}s   "
          f"diffusion {r['t_dif']:.1f}s")
    print(f"inference per action:       regressor {r['t_reg_inf'] * 1e6:.1f} us   "
          f"diffusion {r['t_dif_inf'] * 1e6:.1f} us   "
          f"({K} denoising steps)")
    print()
    print(f"at observation = {r['obs_at']}, where every demonstrator swerved "
          f"about {np.abs(r['truth'][:, 0]).mean():.2f} to one side or the other:")
    print(f"{'':<12}{'left':>9}{'straight':>10}{'right':>9}   mean sideways")
    for name, a in (("demos", r["truth"]), ("regressor", r["reg"]), ("diffusion", r["dif"])):
        left, mid, right = side_fractions(a)
        print(f"{name:<12}{left:>9.0%}{mid:>10.0%}{right:>9.0%}   {a[:, 0].mean():+.3f}")
    print()
    print("'straight' is the answer no demonstrator ever gave.")


if __name__ == "__main__":
    r = run()
    report(r)
    if "--sweep" in sys.argv:
        step_sweep(r["dif_net"])
