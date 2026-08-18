"""Lesson 3.8 - train ACT-lite on real SO-101 demonstrations.

The default dataset is `lerobot/svla_so101_pickplace`: 50 episodes, 11,939
frames at 30 fps, recorded on the same 6-joint arm Module 4 uses. It is read
through `demos.py` from Lesson 3.1, so `--path` points the same run at the
recording you made in Module 2 and nothing else changes.

The defaults are a REDUCED ACT, sized so the run finishes on a laptop CPU:
d_model 128 instead of 512, 2 transformer layers instead of 4, a 20-step chunk
instead of 100, and no cameras. The lesson says what that costs you.

Run:  python train_act_lite.py                      ACT-lite, 5 epochs
      python train_act_lite.py --model mlp          the plain-MLP baseline
      python train_act_lite.py --chunk 1            the no-chunking ablation
      python train_act_lite.py --epochs 1           a smoke test
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import demos
from act_lite import ACTLite, act_loss


# ------------------------------------------------------------------- data

def load_episodes(path=None, repo_id=demos.DEFAULT_REPO):
    """Split the flat frame table back into per-episode (states, actions) pairs.

    Episode boundaries matter here in a way they did not in Lesson 3.1: a chunk
    must never straddle two demonstrations, or the model learns to predict the
    start of the next attempt as the continuation of this one.
    """
    data = demos.load(path, repo_id)
    episodes = [(data.state[data.episode == e], data.action[data.episode == e])
                for e in data.episode_ids]
    return episodes, data.fps


class ChunkDataset(torch.utils.data.Dataset):
    """One sample = the state at t, the next `chunk` actions, and a padding mask.

    Near the end of an episode there are fewer than `chunk` actions left. ACT
    pads with the last action and masks the padded steps out of the loss, which
    is what LeRobot does; the alternative, dropping those samples, throws away
    exactly the part of the task where the robot is finishing the job.
    """

    def __init__(self, episodes, chunk, stats):
        self.chunk = chunk
        self.stats = stats
        self.index = [(e, t) for e, (s, _) in enumerate(episodes)
                      for t in range(len(s))]
        self.episodes = episodes

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        e, t = self.index[i]
        states, actions = self.episodes[e]
        take = min(self.chunk, len(actions) - t)
        window = actions[t:t + take]
        mask = np.zeros(self.chunk, dtype=np.float32)
        mask[:take] = 1.0
        if take < self.chunk:
            window = np.concatenate([window,
                                     np.repeat(window[-1:], self.chunk - take, 0)])
        s_mean, s_std, a_mean, a_std = self.stats
        return (torch.from_numpy((states[t] - s_mean) / s_std),
                torch.from_numpy((window - a_mean) / a_std),
                torch.from_numpy(mask))


def compute_stats(episodes):
    """Per-joint mean and standard deviation. LeRobot keeps these in meta/stats.json."""
    S = np.concatenate([s for s, _ in episodes])
    A = np.concatenate([a for _, a in episodes])
    return (S.mean(0), S.std(0) + 1e-6, A.mean(0), A.std(0) + 1e-6)


# ------------------------------------------------------------- the baseline

class MLPPolicy(nn.Module):
    """Per-step behaviour cloning, reshaped to emit a chunk so the two are comparable."""

    def __init__(self, state_dim=6, action_dim=6, chunk_size=50, hidden=512):
        super().__init__()
        self.chunk_size, self.action_dim = chunk_size, action_dim
        self.net = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, chunk_size * action_dim))

    def forward(self, state, action_chunk=None):
        out = self.net(state).view(-1, self.chunk_size, self.action_dim)
        return out, None, None


# ---------------------------------------------------------------- training

def train(args):
    torch.manual_seed(args.seed)
    episodes, fps = load_episodes(args.path, args.repo_id)
    n_val = max(1, len(episodes) // 10)
    val_eps, train_eps = episodes[:n_val], episodes[n_val:]
    stats = compute_stats(train_eps)

    train_set = ChunkDataset(train_eps, args.chunk, stats)
    val_set = ChunkDataset(val_eps, args.chunk, stats)
    loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch,
                                         shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=256)

    if args.model == "act":
        model = ACTLite(chunk_size=args.chunk, d_model=args.d_model,
                        dim_feedforward=args.ff, n_encoder_layers=args.layers,
                        n_vae_encoder_layers=args.layers)
    else:
        model = MLPPolicy(chunk_size=args.chunk)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    steps_per_epoch = len(loader)
    print(f"{args.model} | {n_params:,} params | chunk {args.chunk} | "
          f"{len(train_eps)} train / {len(val_eps)} val episodes | "
          f"{len(train_set):,} samples | {steps_per_epoch} steps/epoch "
          f"at batch {args.batch}")

    history = []
    t_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        run = 0.0
        for state, chunk, mask in loader:
            pred, mu, logvar = model(state, chunk)
            loss, recon, kl = act_loss(pred, chunk, mu, logvar,
                                       kl_weight=args.kl_weight, mask=mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += recon.item()
        train_l1 = run / steps_per_epoch
        val_l1 = evaluate(model, val_loader)
        history.append({"epoch": epoch + 1, "train_l1": train_l1, "val_l1": val_l1,
                        "seconds": time.time() - t_start})
        print(f"  epoch {epoch+1:>2}  train L1 {train_l1:.4f}  val L1 {val_l1:.4f}"
              f"   {time.time()-t_start:6.1f} s")

    wall = time.time() - t_start
    per_index = per_chunk_index_error(model, val_loader)
    print(f"done in {wall:.1f} s "
          f"({wall/(args.epochs*steps_per_epoch)*1000:.0f} ms per optimizer step)")
    print(f"val L1 at chunk index 0: {per_index[0]:.4f}   "
          f"at index {len(per_index)-1}: {per_index[-1]:.4f}")

    out = {"model": args.model, "chunk": args.chunk, "params": n_params,
           "batch": args.batch, "epochs": args.epochs, "lr": args.lr,
           "steps_per_epoch": steps_per_epoch, "wall_seconds": wall,
           "history": history, "per_chunk_index_l1": per_index}
    path = Path(__file__).with_name(f"train_{args.model}_chunk{args.chunk}.json")
    path.write_text(json.dumps(out))
    print(f"wrote {path}")
    return out


@torch.no_grad()
def evaluate(model, loader):
    """Mean L1 over unpadded steps, in normalised units."""
    model.eval()
    total, count = 0.0, 0.0
    for state, chunk, mask in loader:
        pred, _, _ = model(state)
        per_step = (pred - chunk).abs().mean(dim=-1)
        total += (per_step * mask).sum().item()
        count += mask.sum().item()
    return total / count


@torch.no_grad()
def per_chunk_index_error(model, loader):
    """Mean L1 for each position in the chunk: how far ahead can it see?"""
    model.eval()
    total = count = None
    for state, chunk, mask in loader:
        pred, _, _ = model(state)
        per_step = (pred - chunk).abs().mean(dim=-1)
        s = (per_step * mask).sum(dim=0)
        c = mask.sum(dim=0)
        total = s if total is None else total + s
        count = c if count is None else count + c
    return (total / count.clamp(min=1)).tolist()


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["act", "mlp"], default="act")
    p.add_argument("--path", help="a directory written by Module 2's recorder")
    p.add_argument("--repo-id", default=demos.DEFAULT_REPO)
    p.add_argument("--chunk", type=int, default=20)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--kl-weight", type=float, default=10.0)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--ff", type=int, default=512)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    train(parse())
