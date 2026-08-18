"""Lesson 3.11 - three policy heads, one dataset, one budget, real rollouts.

The task: drive a point from the start to the goal past an obstacle. Every
demonstration goes round it. Which side is a coin flip that the observation
never reveals, so at the start state the demonstrated action distribution has
two lumps and nothing in the middle.

Three heads, identical data, identical gradient-step budget, roughly matched
parameter counts:

  mlp        regress the action chunk under mean squared error
  cvae       ACT's head: a latent variable soaks up which mode a demo was in
  diffusion  Diffusion Policy's head: denoise a chunk out of pure noise

All three are chunked: predict H actions, execute all H, then look again.

The control run (--control) trains the same three on demonstrations that always
go the same way round. There, the distribution has one lump, and the failure
this lesson is about disappears.

Run:  python compare_policies.py              the comparison
      python compare_policies.py --control    the unimodal control
"""
import sys
import time

import numpy as np
import torch
import torch.nn as nn

SEED = 0
H = 8                      # action chunk length
T = 40                     # steps per episode
K = 100                    # denoising steps
GOAL = np.array([0.0, 1.6])
OBST = np.array([0.0, 0.8])
R_OBST = 0.30
STEP_MAX = 0.08
Z_DIM = 8


# --------------------------------------------------------------------------
# the environment, vectorised over episodes

def rollout(policy, n, seed=SEED, chunk=H):
    """Run n episodes at once. Returns (success, collided, paths)."""
    rng = np.random.default_rng(seed)
    pos = np.stack([rng.normal(0, 0.02, n), np.zeros(n)], axis=1)
    alive = np.ones(n, dtype=bool)
    hit = np.zeros(n, dtype=bool)
    done = np.zeros(n, dtype=bool)
    paths = [pos.copy()]

    for start in range(0, T, chunk):
        actions = policy(pos.astype(np.float32))            # (n, H, 2)
        for j in range(min(chunk, T - start)):
            step = actions[:, j, :]
            norm = np.linalg.norm(step, axis=1, keepdims=True)
            step = step * np.minimum(1.0, STEP_MAX / np.maximum(norm, 1e-9))
            pos = np.where(alive[:, None], pos + step, pos)
            hit |= alive & (np.linalg.norm(pos - OBST, axis=1) < R_OBST)
            done |= alive & (np.linalg.norm(pos - GOAL, axis=1) < 0.10)
            alive &= ~(hit | done)
            paths.append(pos.copy())

    return done & ~hit, hit, np.stack(paths, axis=1)


def expert_path(side, amp, rng):
    """A smooth arc from the start to the goal, passing on the given side.

    The warp and the jitter are there because no two human demonstrations are
    identical: some drivers accelerate through the middle, and nobody's hand is
    smooth. Without them every forward step would be the same number, which is
    not a dataset any real robot ever produces.
    """
    t = np.linspace(0.0, 1.0, T + 1) ** rng.uniform(0.90, 1.10)
    x = side * amp * np.sin(np.pi * t) + rng.normal(0, 0.02)
    y = GOAL[1] * t
    return np.stack([x, y], axis=1) + rng.normal(0, 0.004, (T + 1, 2))


def demonstrations(n_ep, seed=SEED, one_sided=False):
    """(obs, chunks): what the demonstrator saw, and the next H actions it took."""
    rng = np.random.default_rng(seed)
    obs, chunks = [], []
    for _ in range(n_ep):
        side = 1.0 if one_sided else rng.choice([-1.0, 1.0])
        path = expert_path(side, rng.uniform(0.42, 0.58), rng)
        acts = np.diff(path, axis=0)                        # (T, 2)
        acts = np.concatenate([acts, np.repeat(acts[-1:], H, axis=0)], axis=0)
        for k in range(T):
            obs.append(path[k])
            chunks.append(acts[k:k + H])
    return (np.asarray(obs, dtype=np.float32),
            np.asarray(chunks, dtype=np.float32).reshape(len(obs), H * 2))


# --------------------------------------------------------------------------
# normalisation, which is not optional
#
# Actions here are steps of about 0.045 units. A diffusion model mixes in noise
# of unit scale, so unnormalised the signal is drowned before step 10 and the
# network has nothing to learn from. LeRobot normalises with the statistics in
# meta/stats.json for exactly this reason. Every head gets the same treatment,
# so the comparison stays fair.

class Norm:
    def __init__(self, x, floor=1e-3):
        self.mu = x.mean(axis=0)
        self.sigma = np.maximum(x.std(axis=0), floor)

    def to(self, x):
        return ((x - self.mu) / self.sigma).astype(np.float32)

    def back(self, x):
        return x * torch.as_tensor(self.sigma, dtype=torch.float32) + \
            torch.as_tensor(self.mu, dtype=torch.float32)


# --------------------------------------------------------------------------
# the three heads

def mlp(d_in, d_out, width=256, depth=3):
    layers, d = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.SiLU()]
        d = width
    return nn.Sequential(*layers, nn.Linear(width, d_out))


class Cvae(nn.Module):
    """ACT's head, shrunk. The encoder sees the answer; the decoder does not."""

    def __init__(self, d_obs=2, d_act=None, z=Z_DIM, width=256):
        super().__init__()
        # Read H now rather than at import, so changing the chunk length at the
        # top of the file is all the chunk-length sweep needs.
        d_act = d_act or H * 2
        self.enc = mlp(d_obs + d_act, 2 * z, width, depth=2)
        self.dec = mlp(d_obs + z, d_act, width, depth=3)
        self.z = z

    def forward(self, obs, chunk):
        mu, logvar = self.enc(torch.cat([obs, chunk], -1)).chunk(2, dim=-1)
        std = (0.5 * logvar).exp()
        z = mu + std * torch.randn_like(std)
        kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(-1).mean()
        return self.dec(torch.cat([obs, z], -1)), kl

    @torch.no_grad()
    def act(self, obs, sample_z=False):
        z = torch.randn(len(obs), self.z) if sample_z else torch.zeros(len(obs), self.z)
        return self.dec(torch.cat([obs, z], -1))


def schedule(k_steps=K, lo=1e-4, hi=0.02):
    betas = torch.linspace(lo, hi, k_steps)
    alphas = 1.0 - betas
    return betas, alphas, torch.cumprod(alphas, dim=0)


class Diffusion(nn.Module):
    """Diffusion Policy's head, shrunk: predict the noise on an action chunk."""

    def __init__(self, d_obs=2, d_act=None, width=256):
        super().__init__()
        self.net = mlp((d_act or H * 2) + 2 + d_obs, d_act or H * 2, width, depth=3)

    def forward(self, chunk_k, k, obs):
        t = k.float().unsqueeze(-1) / K
        emb = torch.cat([torch.sin(2 * np.pi * t), torch.cos(2 * np.pi * t)], -1)
        return self.net(torch.cat([chunk_k, emb, obs], -1))

    @torch.no_grad()
    def act(self, obs, generator=None, steps=K):
        """Denoise a chunk. `steps` < K walks a subsequence, deterministically.

        The full-length loop is the ancestral sampler from the lesson. Skipping
        steps needs the deterministic update instead, because the ancestral
        noise term is only calibrated for single steps.
        """
        betas, alphas, abar = schedule()
        ks = list(range(0, K, max(1, K // steps)))
        a = torch.randn((len(obs), H * 2), generator=generator)
        for i, k in reversed(list(enumerate(ks))):
            kk = torch.full((len(obs),), k, dtype=torch.long)
            eps = self(a, kk, obs)
            if len(ks) == K:
                a = (a - betas[k] / (1 - abar[k]).sqrt() * eps) / alphas[k].sqrt()
                if k > 0:
                    a = a + betas[k].sqrt() * torch.randn(a.shape, generator=generator)
            else:
                a0 = (a - (1 - abar[k]).sqrt() * eps) / abar[k].sqrt()
                prev = abar[ks[i - 1]] if i else torch.tensor(1.0)
                a = prev.sqrt() * a0 + (1 - prev).sqrt() * eps
        return a


# --------------------------------------------------------------------------
# training, one budget for all three

def train(kind, obs, chunks, steps=2500, batch=256, lr=1e-3, seed=SEED, kl_weight=10.0):
    torch.manual_seed(seed)
    norm = Norm(chunks)
    o = torch.as_tensor(obs)
    c = torch.as_tensor(norm.to(chunks))
    g = torch.Generator().manual_seed(seed)
    _, _, abar = schedule()

    if kind == "mlp":
        net = mlp(2, H * 2)
    elif kind == "cvae":
        net = Cvae()
    else:
        net = Diffusion()
    opt = torch.optim.AdamW(net.parameters(), lr=lr)

    t0 = time.time()
    for _ in range(steps):
        i = torch.randint(0, len(o), (batch,), generator=g)
        ob, ch = o[i], c[i]
        if kind == "mlp":
            loss = ((net(ob) - ch) ** 2).mean()
        elif kind == "cvae":
            recon, kl = net(ob, ch)
            loss = ((recon - ch) ** 2).mean() + kl_weight * kl / (H * 2)
        else:
            k = torch.randint(0, K, (batch,), generator=g)
            eps = torch.randn(ch.shape, generator=g)
            ck = abar[k].sqrt().unsqueeze(-1) * ch + (1 - abar[k]).sqrt().unsqueeze(-1) * eps
            loss = ((net(ck, k, ob) - eps) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    n_params = sum(p.numel() for p in net.parameters())
    return net, norm, time.time() - t0, n_params


def head_output(kind, net, obs, generator=None, sample_z=False, steps=K):
    """One forward pass of whichever head, in normalised action units."""
    if kind == "mlp":
        return net(obs)
    if kind == "cvae":
        return net.act(obs, sample_z=sample_z)
    return net.act(obs, generator=generator, steps=steps)


def make_policy(kind, net, norm, seed=SEED, sample_z=False, steps=K):
    """Wrap a trained net as obs -> (n, H, 2) actions, for the rollout loop."""
    gen = torch.Generator().manual_seed(seed)

    def policy(obs):
        ob = torch.as_tensor(obs)
        with torch.no_grad():
            out = norm.back(head_output(kind, net, ob, gen, sample_z, steps))
        return out.numpy().reshape(len(obs), H, 2)
    return policy


# --------------------------------------------------------------------------
# honest reporting

def wilson(successes, n, z=1.96):
    """95% confidence interval for a success rate. Lesson 3.12 lives here."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


@torch.no_grad()
def val_mse(kind, net, norm, obs, chunks, sample_z=False):
    """Mean squared error against the held-out demonstrated chunk, in real units."""
    ob = torch.as_tensor(obs)
    pred = norm.back(head_output(kind, net, ob, torch.Generator().manual_seed(7),
                                 sample_z))
    return float(((pred.numpy() - chunks) ** 2).mean())


def evaluate(one_sided=False, steps=2500, n_eval=200, seed=SEED):
    torch.set_num_threads(2)
    obs, chunks = demonstrations(50, seed=seed, one_sided=one_sided)
    obs_v, chunks_v = demonstrations(20, seed=seed + 100, one_sided=one_sided)

    rows, paths = [], {}
    variants = [("mlp", "mlp", {}), ("cvae", "cvae", {}),
                ("cvae", "cvae (z sampled)", dict(sample_z=True)),
                ("diffusion", "diffusion", {})]

    trained = {}
    for kind in ("mlp", "cvae", "diffusion"):
        trained[kind] = train(kind, obs, chunks, steps=steps, seed=seed)

    for kind, label, kw in variants:
        net, norm, secs, n_params = trained[kind]
        policy = make_policy(kind, net, norm, seed=seed + 5, **kw)

        t0 = time.time()
        ok, hit, path = rollout(policy, n_eval, seed=seed + 7)
        infer_ms = 1000 * (time.time() - t0) / (n_eval * np.ceil(T / H))

        lo, hi = wilson(int(ok.sum()), n_eval)
        rows.append(dict(policy=label, params=n_params, train_s=secs,
                         mse=val_mse(kind, net, norm, obs_v, chunks_v, **kw),
                         success=int(ok.sum()), n=n_eval, lo=lo, hi=hi,
                         collided=int(hit.sum()), infer_ms=infer_ms))
        paths[label] = path
    return rows, paths, (obs, chunks)


def report(rows, title):
    print(f"\n{title}")
    print(f"{'policy':<18}{'params':>9}{'train s':>9}{'val MSE':>11}"
          f"{'success':>10}{'95% CI':>16}{'hit obstacle':>14}{'ms/chunk':>10}")
    for r in rows:
        ci = f"{r['lo']:.0%}-{r['hi']:.0%}"
        print(f"{r['policy']:<18}{r['params']:>9,}{r['train_s']:>9.1f}{r['mse']:>11.5f}"
              f"{r['success']:>6}/{r['n']:<3}{ci:>16}{r['collided']:>14}{r['infer_ms']:>10.2f}")


if __name__ == "__main__":
    control = "--control" in sys.argv
    rows, paths, data = evaluate(one_sided=control)
    report(rows, "demonstrations always go the same way (unimodal control)" if control
           else "demonstrations go either way (multimodal)")
