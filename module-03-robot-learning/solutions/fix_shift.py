"""Solution - lesson 3.5, four fixes at a matched label budget.

Every method here gets the SAME budget: 3,000 (state, action) pairs labelled by
the expert. That is the only fair comparison, because expert labels are the
thing that costs money in the real world.

  plain BC     50 demonstrations, all starting from home
  noise        50 demonstrations, with the executed action disturbed
  recovery     50 demonstrations, half of them started off centre
  late         50 demonstrations, half shoved off centre partway through
  DAgger       20 demonstrations, then the expert labels states the CLONE visits

Run:  python fix_shift.py             all five, one seed
      python fix_shift.py --seeds 3   all five, three seeds, with spread
      python fix_shift.py --trace     DAgger round by round
      python fix_shift.py --where     where each fix put its corrective frames
"""
import argparse

import numpy as np

from groove_world import (DEMO_STEPS, EVAL_STEPS, PATH, collect, dls, evaluate,
                          expert, fit_policy, fk, nearest_on_path, rollout,
                          start_pose)

EPOCHS = 200
BUDGET = 50 * DEMO_STEPS          # 3,000 expert-labelled pairs


def sideways(theta, distance, rng):
    """Shove the pin `distance` metres across the groove, either way."""
    tangent = PATH[6] - PATH[0]
    tangent = tangent / np.linalg.norm(tangent)
    normal = np.array([-tangent[1], tangent[0]])
    sign = 1.0 if rng.random() < 0.5 else -1.0
    return theta + dls(theta, sign * distance * normal)


def collect_late(n_episodes, seed, offset=0.06):
    """Recovery demonstrations where the shove lands partway through the episode.

    `collect(..., recovery=)` starts the arm off centre and lets the expert drive
    back, which is what "recovery demonstration" usually means. Every corrective
    frame it produces therefore sits at the beginning of the groove. This
    version puts the shove somewhere random in the middle instead. Same expert,
    same number of labels, different place on the path.
    """
    rng = np.random.default_rng(seed)
    states, actions = [], []
    for k in range(n_episodes):
        theta = start_pose(rng)
        shove_at = rng.integers(10, DEMO_STEPS) if k % 2 else None
        for t in range(DEMO_STEPS):
            action = expert(theta)
            states.append(theta.copy())
            actions.append(action)
            theta = theta + action
            if t == shove_at:
                theta = sideways(theta, offset * rng.uniform(0.3, 1.0), rng)
    return np.array(states), np.array(actions)


def dagger(seed, seed_demos=20, rounds=5, per_round=3, rollout_len=120,
           epochs=EPOCHS, trace=False):
    """Roll out the clone, have the expert label where it went, retrain, repeat.

    The label budget is spent the same way as everywhere else in this file:
    20 demonstrations is 1,200 pairs, then five rounds of three 120-step
    rollouts adds 1,800 more, for 3,000 in total.
    """
    states, actions = collect(seed_demos, seed)
    rng = np.random.default_rng(1000 + seed)
    history = []
    for round_index in range(rounds + 1):
        policy, _ = fit_policy(states, actions, epochs=epochs, seed=seed)
        if trace:
            scored = evaluate(policy, steps=EVAL_STEPS)
            history.append((len(states), scored["success"], scored["survived"]))
            print(f"  round {round_index}: {len(states):5d} labels   "
                  f"success {scored['success']:.2f}   survived {scored['survived']:.0f}")
        if round_index == rounds:
            return policy, history

        visited = np.concatenate([
            rollout(policy, start_pose(rng), steps=rollout_len, walls=False)["states"]
            for _ in range(per_round)])
        labels = np.array([expert(state) for state in visited])
        states = np.concatenate([states, visited])
        actions = np.concatenate([actions, labels])


def datasets(seed=0):
    """The four training sets the lesson compares, built the same way `run` does."""
    return {
        "plain demonstrations": collect(50, seed),
        "recovery at the start": collect(50, seed, recovery=0.06),
        "noise injection": collect(50, seed, noise=0.02),
        "late shove": collect_late(50, seed),
    }


def where_the_data_is(seed=0, n_fail=30):
    """Why recovery demonstrations do nothing: they land in the wrong place.

    Two measurements, and the second is the one that settles it.

    First, count the rows that sit more than a centimetre off the groove centre
    - the corrective frames - and ask where along the path from A to B they are.
    Recovery demonstrations put theirs in the first few percent, because a
    competent expert is back on the path within a handful of steps.

    Second, roll the plain clone out until it jams, and measure how far its
    failure states are from the nearest row of each training set. A fix that
    does not shrink that distance has not bought anything.
    """
    sets = datasets(seed)

    print("rows more than 1 cm off the groove centre, and where they sit on the path")
    for name, (states, _) in sets.items():
        located = [nearest_on_path(fk(state)) for state in states]
        off = np.array([i / (len(PATH) - 1) for i, gap in located if gap > 0.01])
        if len(off):
            print(f"  {name:24s} {len(off):5d} rows   "
                  f"{off.mean():.2f} average, {np.percentile(off, 90):.2f} at the 90th percentile")
        else:
            print(f"  {name:24s} {0:5d} rows")

    states, actions = sets["plain demonstrations"]
    policy, _ = fit_policy(states, actions, epochs=EPOCHS, seed=seed)
    rng = np.random.default_rng(11)
    failures = np.array([
        rollout(policy, start_pose(rng), steps=EVAL_STEPS)["states"][-1]
        for _ in range(n_fail)])

    print("\ndistance from the clone's failure states to the nearest row of each set")
    for name, (rows, _) in sets.items():
        gaps = np.linalg.norm(failures[:, None, :] - rows[None, :, :], axis=2)
        print(f"  {name:24s} {gaps.min(axis=1).mean():.4f}")


def method(name, seed, collector=collect, **kw):
    states, actions = collector(50, seed, **kw)
    policy, _ = fit_policy(states, actions, epochs=EPOCHS, seed=seed)
    return name, policy, len(states)


def run(seeds=1):
    rows = {}
    for seed in range(seeds):
        built = [method("plain BC", seed),
                 method("noise 0.02", seed, noise=0.02),
                 method("noise 0.05", seed, noise=0.05),
                 method("recovery", seed, recovery=0.06),
                 method("late shove", seed, collector=collect_late)]
        policy, _ = dagger(seed)
        built.append(("DAgger", policy, BUDGET))
        for name, policy, labels in built:
            nominal = evaluate(policy, steps=EVAL_STEPS)
            nudged = evaluate(policy, steps=EVAL_STEPS, offset=0.04)
            rows.setdefault(name, []).append(
                (nominal["success"], nominal["survived"], nudged["success"], labels))

    print(f"\n{'method':12s} {'labels':>7s}  {'success @240':>14s}  "
          f"{'survived':>9s}  {'success, 4 cm off':>18s}")
    baseline = evaluate(expert, steps=EVAL_STEPS)
    print(f"{'expert':12s} {'-':>7s}  {baseline['success']:14.2f}  "
          f"{baseline['survived']:9.0f}  {evaluate(expert, steps=EVAL_STEPS, offset=0.04)['success']:18.2f}")
    for name, values in rows.items():
        v = np.array(values)
        spread = f" +- {v[:, 0].std():.2f}" if seeds > 1 else ""
        print(f"{name:12s} {v[0, 3]:7.0f}  {v[:, 0].mean():9.2f}{spread:>5s}  "
              f"{v[:, 1].mean():9.0f}  {v[:, 2].mean():18.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--where", action="store_true")
    args = parser.parse_args()
    if args.trace:
        dagger(0, trace=True)
    elif args.where:
        where_the_data_is()
    else:
        run(args.seeds)
