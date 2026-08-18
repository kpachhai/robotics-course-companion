"""Lesson 3.2 - a policy is a function from observation to action. Build four.

No training, no torch, no gradients. Every policy here is a plain Python object
with one method, `act(observation) -> action`, and the whole point is that this
is already the entire interface a robot ever asks of a learned controller.

  DictPolicy      look the observation up in a table of what the human did
  NearestPolicy   no exact match? use the closest observation you have seen
  ConstantPolicy  always command the average action (the do-nothing baseline)
  EchoPolicy      command whatever you are already at (the hold-still baseline)

Three TODOs. When they pass you will have measured something uncomfortable:
the policy with the lowest error is the one that never moves.

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
        """Return the recorded action for this exact observation, or None.

        Count a hit or a miss on every call - the miss rate is the measurement
        this whole class exists to produce.
        """
        # TODO(you): look up tuple(observation.tolist()), update self.hits /
        # self.misses, and return the action or None.
        raise NotImplementedError


class NearestPolicy:
    """Same table, but the closest key wins instead of an exact one.

    `scale` divides each dimension before distances are measured. Pass the
    per-joint standard deviation and a wrist wobble stops being invisible next
    to a shoulder sweep.
    """

    def __init__(self, states, actions, scale=None):
        self.scale = np.ones(states.shape[1]) if scale is None else np.asarray(scale)
        self.keys = states / self.scale
        self.actions = actions

    def act(self, observation):
        """Return the action recorded beside the nearest stored observation."""
        # TODO(you): scale the observation, find the row of self.keys with the
        # smallest squared distance to it, return that row's action.
        raise NotImplementedError


class ConstantPolicy:
    """Given complete. Ignores the observation entirely."""

    def __init__(self, actions):
        self.action = actions.mean(0)

    def act(self, observation):
        return self.action


class EchoPolicy:
    """Given complete. Commands the joint angles it is already at.

    A real robot running this does nothing at all: every command is the pose it
    is already holding. Remember that when you read its error.
    """

    def act(self, observation):
        return observation


def evaluate(policy, states, actions, fallback=None):
    """Mean absolute error between commanded and recorded action, in units.

    One unit is roughly one percent of that joint's calibrated travel. `fallback`
    is the action to use when a policy returns None, so DictPolicy can be scored
    at all.
    """
    # TODO(you): call policy.act on each row of `states`, substitute `fallback`
    # for any None, and return the mean of |predicted - actual| over every
    # frame and every joint.
    raise NotImplementedError


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
