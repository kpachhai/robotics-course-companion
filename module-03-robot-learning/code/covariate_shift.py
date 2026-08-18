"""Lesson 3.4 - the clone works for sixty steps. Watch what it does for 240.

`groove_world.py` is given to you complete: a 2-link arm, a curved groove, and a
scripted expert that never fails. This file clones the expert and then measures
the one thing a held-out loss cannot see.

Three TODOs.

Run:  python covariate_shift.py            train, then the two horizons
      python covariate_shift.py --probe    same network, two state distributions
      python covariate_shift.py --drift    how far off the data it gets
      python covariate_shift.py --demos    does more data fix it?
"""
import argparse

import numpy as np

from groove_world import (DEMO_STEPS, EVAL_STEPS, collect, evaluate, expert,
                          fit_policy, rollout, start_pose)

DEMOS = 50
EPOCHS = 200


def clone(n_demos=DEMOS, seed=0, epochs=EPOCHS, **kw):
    states, actions = collect(n_demos, seed, **kw)
    policy, history = fit_policy(states, actions, epochs=epochs, seed=seed)
    return policy, history, states


def two_horizons(policy):
    """The whole lesson in six numbers.

    Nothing changes between these two rows except how long we let the policy run.
    Sixty steps is the length of a demonstration. 240 steps is what a real robot
    does, which is run until somebody stops it.
    """
    for steps in (DEMO_STEPS, EVAL_STEPS):
        # TODO(you): call evaluate(policy, steps=steps) for the clone and for
        # `expert`, and print success rate plus median steps survived for both.
        raise NotImplementedError


def probe(policy, n=30, seed=7):
    """Same network, two state distributions. This is covariate shift, isolated.

    Feed the clone the states the EXPERT visits and measure how wrong its action
    is. Then feed it the states IT visits and measure the same thing. One
    network, one task, one metric - only the states differ.
    """
    rng = np.random.default_rng(seed)
    on_expert, on_own = [], []
    for _ in range(n):
        theta0 = start_pose(rng)
        expert_run = rollout(expert, theta0, steps=DEMO_STEPS, walls=False)
        clone_run = rollout(policy, theta0, steps=DEMO_STEPS, walls=False)
        # TODO(you): for each state in expert_run["states"], append
        # ||policy(state) - expert(state)|| to on_expert. Do the same over
        # clone_run["states"] into on_own.
        raise NotImplementedError
    on_expert, on_own = np.array(on_expert), np.array(on_own)
    print(f"  action error on the expert's states   {on_expert.mean():.6f} rad")
    print(f"  action error on its own states        {on_own.mean():.6f} rad")
    print(f"  over the last ten steps               "
          f"{on_expert[:, -10:].mean():.6f} vs {on_own[:, -10:].mean():.6f} rad "
          f"({on_own[:, -10:].mean() / on_expert[:, -10:].mean():.0f}x)")
    return on_expert, on_own


def drift(policy, train_states, n=30, seed=11):
    """Distance from the states a policy visits to the nearest state it trained on.

    This is the definition of the problem, plotted. Zero means "I have seen this
    before". Growing means the policy is writing its own test set.
    """
    rng = np.random.default_rng(seed)
    expert_d, clone_d = [], []
    for _ in range(n):
        theta0 = start_pose(rng)
        for run, bucket in ((rollout(expert, theta0, steps=EVAL_STEPS, walls=False), expert_d),
                            (rollout(policy, theta0, steps=EVAL_STEPS, walls=False), clone_d)):
            # TODO(you): for every state in run["states"], find the distance to
            # the closest row of `train_states` and collect the whole sequence.
            # scipy is not needed: broadcasting over 3,000 rows is fast enough.
            raise NotImplementedError
    expert_d, clone_d = np.array(expert_d), np.array(clone_d)
    print("  step   expert    clone   (distance to the nearest training state, rad)")
    for t in (0, 30, 60, 90, 120, 180, EVAL_STEPS - 1):
        print(f"  {t:4d}   {expert_d[:, t].mean():.5f}   {clone_d[:, t].mean():10.5f}")
    return expert_d, clone_d


def demo_sweep(seeds=3):
    print("demos   held-out loss   success @240   median steps survived")
    for n_demos in (10, 25, 50, 100, 200):
        scores, survived, losses = [], [], []
        for seed in range(seeds):
            policy, history, _ = clone(n_demos, seed)
            result = evaluate(policy, steps=EVAL_STEPS)
            scores.append(result["success"])
            survived.append(result["survived"])
            losses.append(history[-1, 1])
        print(f"{n_demos:5d}   {np.mean(losses):13.5f}   "
              f"{np.mean(scores):5.2f} +- {np.std(scores):.2f}   {np.mean(survived):6.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--drift", action="store_true")
    parser.add_argument("--demos", action="store_true")
    args = parser.parse_args()

    if args.demos:
        demo_sweep()
    else:
        policy, history, states = clone()
        print(f"cloned {DEMOS} demonstrations, held-out loss {history[-1, 1]:.6f}\n")
        if args.probe:
            probe(policy)
        elif args.drift:
            drift(policy, states)
        else:
            two_horizons(policy)
