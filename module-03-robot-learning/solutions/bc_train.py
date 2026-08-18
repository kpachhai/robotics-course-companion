"""Solution - lesson 3.3, behaviour cloning on the SO-101 demonstrations.

Same dataset and same split as lesson 3.2, so the numbers land in the same table.
Measured on the 50-episode Hub dataset, 40 train / 10 held out: do nothing
3.020, nearest neighbour 4.884, this 2.108.

Run:  python bc_train.py                     train and score
      python bc_train.py --capacity          does a bigger network help?
      python bc_train.py --path pick_demos   your own Module 2 recording
"""
import argparse
import time

import numpy as np
import torch
from torch import nn

from demos import DEFAULT_REPO, load

N_TRAIN = 40          # episodes used for fitting; the rest are held out
EPOCHS = 30
BATCH = 128
LR = 1e-3
HIDDEN = 256


def standardise(train, other):
    """Rescale so every column has mean 0 and spread 1 on the TRAINING set.

    Returns (train_scaled, other_scaled, mean, std). The mean and std come from
    the training half only - computing them over the whole dataset would let the
    held-out episodes influence the fit, which is leakage.
    """
    mean = train.mean(0)
    std = train.std(0) + 1e-6
    return (train - mean) / std, (other - mean) / std, mean, std


def build_net(n_in, n_out, hidden=HIDDEN):
    """Two hidden layers, ReLU between them, plain linear output.

    No activation on the last layer: the output is a joint angle, which is free
    to be any real number. A tanh or a sigmoid here would silently clamp the
    arm's reachable range to whatever that function can produce.
    """
    return nn.Sequential(
        nn.Linear(n_in, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, n_out),
    )


def train_epoch(net, opt, X, Y, order):
    """One pass over the training rows in `order`. Returns the mean loss."""
    total = 0.0
    for i in range(0, len(order), BATCH):
        batch = order[i:i + BATCH]
        loss = ((net(X[batch]) - Y[batch]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(batch)
    return total / len(order)


def action_error(net, X_scaled, Y_raw, mean, std):
    """Mean absolute error in the dataset's own units, not scaled ones.

    The network predicts in standardised space, so undo the scaling before
    comparing. Reporting a scaled number is the easiest way to publish an error
    that nobody, including you, can interpret.
    """
    with torch.no_grad():
        predicted = net(X_scaled).numpy() * std + mean
    return float(np.abs(predicted - Y_raw).mean())


def fit(demos, epochs=EPOCHS, hidden=HIDDEN, seed=0, log=True):
    torch.manual_seed(seed)
    train_ids, test_ids = demos.split(N_TRAIN)
    train, test = demos.mask(train_ids), demos.mask(test_ids)
    state_train, action_train = demos.state[train], demos.action[train]
    state_test, action_test = demos.state[test], demos.action[test]

    X, X_test, _, _ = standardise(state_train, state_test)
    Y, _, y_mean, y_std = standardise(action_train, action_train)

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    net = build_net(X.shape[1], Y.shape[1], hidden)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(seed)

    history, started = [], time.time()
    for epoch in range(epochs):
        order = torch.randperm(len(X), generator=gen)
        loss = train_epoch(net, opt, X, Y, order)
        held_out = action_error(net, X_test, action_test, y_mean, y_std)
        history.append((loss, held_out))
        if log and epoch % 5 == 0:
            print(f"  epoch {epoch:3d}   train loss {loss:.4f}   held-out error {held_out:.3f}")
    if log:
        print(f"  {epochs} epochs in {time.time() - started:.1f}s, "
              f"{sum(p.numel() for p in net.parameters()):,} parameters")
    return net, np.array(history), (X_test, action_test, y_mean, y_std)


def report(demos):
    train_ids, test_ids = demos.split(N_TRAIN)
    train, test = demos.mask(train_ids), demos.mask(test_ids)
    state_test, action_test = demos.state[test], demos.action[test]

    print(f"{len(train_ids)} episodes for fitting ({train.sum()} frames), "
          f"{len(test_ids)} held out ({test.sum()} frames)\n")
    net, history, (X_test, _, y_mean, y_std) = fit(demos)

    echo = np.abs(state_test - action_test).mean()
    print(f"\nheld-out action error, lower is better")
    print(f"  do nothing (lesson 3.2)   {echo:.3f}")
    print(f"  behaviour cloning         {history[-1, 1]:.3f}")

    with torch.no_grad():
        predicted = net(X_test).numpy() * y_std + y_mean
    print("\nper joint:")
    for i, name in enumerate(demos.joint_names):
        mine = np.abs(predicted[:, i] - action_test[:, i]).mean()
        theirs = np.abs(state_test[:, i] - action_test[:, i]).mean()
        flag = "  <- worse than doing nothing" if mine > theirs else ""
        print(f"  {name:20s} do nothing {theirs:6.3f}   cloned {mine:6.3f}{flag}")

    one = X_test[:1]
    started = time.perf_counter()
    with torch.no_grad():
        for _ in range(500):
            net(one)
    print(f"\ninference {(time.perf_counter() - started) / 500 * 1e3:.2f} ms/call, "
          f"and it stays there however much data you add")


def capacity(demos):
    print("does a bigger network help?\n")
    for hidden in (32, 64, 128, 256, 512):
        _, history, _ = fit(demos, hidden=hidden, log=False)
        print(f"  hidden {hidden:4d}   held-out error {history[-1, 1]:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="a directory written by Module 2's recorder")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--capacity", action="store_true")
    args = parser.parse_args()
    demos = load(args.path, args.repo)
    (capacity if args.capacity else report)(demos)
