"""Solution - Lesson 3.2: four policies, no training, one uncomfortable result.

Run:  python lookup_policy.py                     the 50-episode SO-101 dataset
      python lookup_policy.py --sweep             error against table size
      python lookup_policy.py --path pick_demos   your own Module 2 recording
"""
import argparse
import time

import numpy as np

from demos import DEFAULT_REPO, Demos, load

N_TRAIN = 40  # episodes used to build the table; the rest are held out


class DictPolicy:
    """The literal lookup table: every observation the human ever saw, keyed."""

    def __init__(self, states, actions):
        self.table = {tuple(s.tolist()): a for s, a in zip(states, actions)}
        self.hits = self.misses = 0

    def act(self, observation):
        found = self.table.get(tuple(observation.tolist()))
        if found is None:
            self.misses += 1
        else:
            self.hits += 1
        return found


class NearestPolicy:
    """Same table, but the closest key wins instead of an exact one."""

    def __init__(self, states, actions, scale=None):
        self.scale = np.ones(states.shape[1]) if scale is None else np.asarray(scale)
        self.keys = states / self.scale
        self.actions = actions

    def act(self, observation):
        gap = self.keys - observation / self.scale
        return self.actions[np.einsum("ij,ij->i", gap, gap).argmin()]


class ConstantPolicy:
    """Ignores the observation entirely."""

    def __init__(self, actions):
        self.action = actions.mean(0)

    def act(self, observation):
        return self.action


class EchoPolicy:
    """Commands the joint angles it is already at, so the arm never moves."""

    def act(self, observation):
        return observation


def evaluate(policy, states, actions, fallback=None):
    """Mean absolute error between commanded and recorded action, in units."""
    predicted = np.empty_like(actions)
    for i, observation in enumerate(states):
        chosen = policy.act(observation)
        predicted[i] = fallback if chosen is None else chosen
    return float(np.abs(predicted - actions).mean())


def report(demos: Demos) -> None:
    train_ids, test_ids = demos.split(N_TRAIN)
    train, test = demos.mask(train_ids), demos.mask(test_ids)
    train_state, train_action = demos.state[train], demos.action[train]
    test_state, test_action = demos.state[test], demos.action[test]
    mean_action = train_action.mean(0)

    print(f"{len(train_ids)} episodes in the table ({train.sum()} frames), "
          f"{len(test_ids)} held out ({test.sum()} frames)")

    exact = DictPolicy(train_state, train_action)
    error = evaluate(exact, test_state, test_action, fallback=mean_action)
    total = exact.hits + exact.misses
    print(f"\nDictPolicy       {exact.hits}/{total} exact hits "
          f"({100 * exact.hits / total:.2f}%), error {error:.3f}")

    nearest = NearestPolicy(train_state, train_action)
    print(f"NearestPolicy    error {evaluate(nearest, test_state, test_action):.3f}")

    scaled = NearestPolicy(train_state, train_action, scale=train_state.std(0))
    print(f"  + std-scaled   error {evaluate(scaled, test_state, test_action):.3f}")

    print(f"ConstantPolicy   error {evaluate(ConstantPolicy(train_action), test_state, test_action):.3f}")
    print(f"EchoPolicy       error {evaluate(EchoPolicy(), test_state, test_action):.3f}"
          "   <- a policy that never moves")

    start = time.perf_counter()
    for observation in test_state[:200]:
        nearest.act(observation)
    per_call = (time.perf_counter() - start) / 200
    print(f"\nNearestPolicy inference {per_call * 1e3:.2f} ms/call over "
          f"{len(train_state)} rows; the control loop has {1000 / demos.fps:.1f} ms")


def sweep(demos: Demos, sizes=(1, 2, 3, 5, 8, 13, 20, 30, 40), repeats=3, seed=0):
    """Error against table size. Random subsets, so the curve is not one anecdote."""
    train_ids, test_ids = demos.split(N_TRAIN)
    test = demos.mask(test_ids)
    test_state, test_action = demos.state[test], demos.action[test]
    rng = np.random.default_rng(seed)

    print("demos   rows    error (mean of "
          f"{repeats} random subsets)   spread")
    rows = []
    for size in sizes:
        scores, counts = [], []
        for _ in range(repeats if size < len(train_ids) else 1):
            chosen = rng.choice(train_ids, size=size, replace=False)
            picked = demos.mask(chosen)
            policy = NearestPolicy(demos.state[picked], demos.action[picked])
            scores.append(evaluate(policy, test_state, test_action))
            counts.append(int(picked.sum()))
        rows.append((size, np.mean(counts), np.mean(scores), np.std(scores)))
        print(f"{size:5d}  {np.mean(counts):6.0f}   {np.mean(scores):6.3f}"
              f"                        {np.std(scores):5.3f}")
    return np.array(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="a directory written by Module 2's recorder")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--sweep", action="store_true", help="error against table size")
    args = parser.parse_args()
    demos = load(args.path, args.repo)
    (sweep if args.sweep else report)(demos)
