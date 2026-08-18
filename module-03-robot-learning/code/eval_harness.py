"""Lesson 3.13 - the evaluation harness you will use for the rest of the course.

Four things are yours: `Protocol.fingerprint`, `paired_schedule`, `summarise`
and `compare`. Everything else - the append-only log, the protocol gate, the
mock robot, the printing - is done.

Run:
    python eval_harness.py                # run the demo evaluation, print the card
    python eval_harness.py --coverage     # why the summary uses Wilson, not Wald
    python eval_harness.py --pairing      # what identical scenes buy you

The harness is deliberately policy-agnostic. Swap `mock_rollout` for a function
that drives your simulator or your arm and nothing else changes.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, norm

Z95 = norm.ppf(0.975)


# ------------------------------------------------------------------ statistics
def wilson_interval(k, n, z=Z95):
    """95% Wilson score interval for k successes out of n (lesson 3.12)."""
    k = np.asarray(k, dtype=float)
    p_hat = k / n
    denom = 1.0 + z * z / n
    centre = (p_hat + z * z / (2 * n)) / denom
    half = z / denom * np.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))
    return float(np.clip(centre - half, 0, 1)), float(np.clip(centre + half, 0, 1))


def wald_interval(k, n, z=Z95):
    """The interval most people write from memory. Kept to show it failing."""
    p_hat = k / n
    half = z * np.sqrt(p_hat * (1 - p_hat) / n)
    return float(np.clip(p_hat - half, 0, 1)), float(np.clip(p_hat + half, 0, 1))


# -------------------------------------------------------------------- protocol
@dataclass(frozen=True)
class Protocol:
    """Everything decided BEFORE the first trial runs.

    Frozen on purpose. Its fingerprint goes in every trial record, so a run
    whose protocol was edited mid-flight cannot be silently appended to.
    """
    task: str
    policies: tuple[str, ...]
    n_trials: int                       # per policy
    seed0: int
    factors: dict                       # name -> [low, high], sampled per trial
    success_rule: str                   # the predicate, in words, versioned here
    failure_tags: tuple[str, ...]
    notes: str = ""

    def fingerprint(self) -> str:
        """Stable short hash of the protocol. Same protocol, same id, forever.

        TODO(you): serialise `asdict(self)` to JSON with `sort_keys=True` and
        `separators=(",", ":")`, then return the first 12 hex characters of its
        SHA-256. Sorting the keys is the load-bearing part: without it, two
        identical protocols can hash differently and the gate below stops working.
        """
        raise NotImplementedError


@dataclass
class Trial:
    protocol_id: str
    policy: str
    index: int                          # pairing key: same index, same scene
    seed: int
    conditions: dict
    success: bool
    failure: str | None
    seconds: float
    stamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))


def paired_schedule(protocol: Protocol):
    """Return (policy, index, seed) in the order to run them.

    Two properties, both load-bearing:
      PAIRED     - trial `index` uses one seed for every policy, so each policy
                   faces the identical initial condition. The comparison is then
                   within-scene, and scene difficulty stops being noise.
      INTERLEAVED - the policy order rotates each index, so anything that drifts
                   over the session (battery, lighting, your own patience)
                   contaminates every policy equally instead of only the last.

    TODO(you): for each index i in range(protocol.n_trials), emit one entry per
    policy with seed `protocol.seed0 + i` (that is the pairing), rotating the
    policy order left by `i % len(policies)` (that is the interleaving).
    The self-check at the bottom of this file tests both properties.
    """
    raise NotImplementedError


def sample_conditions(protocol: Protocol, seed: int) -> dict:
    """The initial condition for a trial, reproducible from its seed alone."""
    rng = np.random.default_rng(seed)
    return {name: float(rng.uniform(lo, hi)) for name, (lo, hi) in protocol.factors.items()}


# ------------------------------------------------------------------- the log
class EvalRun:
    """Append-only JSONL log. The file is the record; nothing is held in memory."""

    def __init__(self, protocol: Protocol, path):
        self.protocol = protocol
        self.path = Path(path)
        self.protocol_id = protocol.fingerprint()

        if self.path.exists():
            header = json.loads(self.path.read_text().splitlines()[0])
            if header["protocol_id"] != self.protocol_id:
                raise ValueError(
                    f"{self.path} was written under protocol {header['protocol_id']}, "
                    f"not {self.protocol_id}. Changing the protocol starts a new file; "
                    "appending to this one would mix two experiments."
                )
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w") as f:
                f.write(json.dumps({"protocol_id": self.protocol_id,
                                    "protocol": asdict(protocol)}) + "\n")

    def record(self, trial: Trial) -> None:
        if trial.protocol_id != self.protocol_id:
            raise ValueError("trial belongs to a different protocol")
        if trial.failure and trial.failure not in self.protocol.failure_tags:
            raise ValueError(f"unknown failure tag {trial.failure!r}; "
                             f"declare it in the protocol first")
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(trial)) + "\n")

    def done(self) -> set:
        """(policy, index) pairs already logged - so a crashed run resumes."""
        return {(t.policy, t.index) for t in self.load(self.path)[1]}

    @staticmethod
    def load(path):
        lines = Path(path).read_text().splitlines()
        header = json.loads(lines[0])
        return header, [Trial(**json.loads(line)) for line in lines[1:]]


# ------------------------------------------------------------------- reporting
def summarise(trials):
    """Per-policy success rate with its Wilson interval and failure histogram.

    TODO(you): return {policy: {"k", "n", "rate", "lo", "hi", "failures"}} where
    "failures" is a `Counter` over the failure tags of the failed trials only.
    Nothing here should ever be typed by hand into a blog post; this function is
    the only place a number is allowed to come from.
    """
    raise NotImplementedError


def compare(trials, policy_a, policy_b):
    """Paired comparison on the scenes both policies actually saw.

    Only the DISAGREEMENTS carry information about which policy is better; the
    scenes both solved and both failed tell you about the task, not the policy.
    That is McNemar's test, and it is why pairing is worth the bookkeeping.

    TODO(you):
      1. group the trials by `index` so you can see both policies on one scene
      2. count the four cells: both succeeded, neither did, A only, B only
      3. the discordant pairs are a_only + b_only. Under "the policies are the
         same", each discordant pair is a fair coin, so the p-value is
         `binomtest(a_only, a_only + b_only, 0.5).pvalue`. With no
         disagreements at all there is nothing to test, so return 1.0.
    Return {"both", "neither", "a_only", "b_only", "n_pairs", "p_value"}.
    """
    raise NotImplementedError


def report(path):
    header, trials = EvalRun.load(path)
    proto = header["protocol"]
    print(f"\n{proto['task']}   protocol {header['protocol_id']}")
    print(f"success rule: {proto['success_rule']}")
    print(f"factors varied: {', '.join(proto['factors'])}\n")

    print(f"  {'policy':<12}{'rate':>10}   95% Wilson        top failure")
    for policy, s in summarise(trials).items():
        worst = s["failures"].most_common(1)
        tag = f"{worst[0][0]} x{worst[0][1]}" if worst else "-"
        print(f"  {policy:<12}{s['k']:>3}/{s['n']:<3}{s['rate']:>5.0%}   "
              f"[{s['lo']:.2f}, {s['hi']:.2f}]   {tag}")

    if len(proto["policies"]) == 2:
        a, b = proto["policies"]
        c = compare(trials, a, b)
        print(f"\n  paired on {c['n_pairs']} identical scenes: "
              f"both {c['both']}, neither {c['neither']}, "
              f"{a} only {c['a_only']}, {b} only {c['b_only']}")
        print(f"  McNemar exact p = {c['p_value']:.3f} on {c['a_only'] + c['b_only']} "
              f"disagreements")


# --------------------------------------------------------------- a fake robot
# Each policy has its own weakness, because that is the whole point of logging
# failure tags: two policies at the same success rate can be failing at
# different things, and only one of those is fixable with the data you have.
WEAKNESS = {
    "act":       {"reach_short": 0.55, "grasp_angle": 0.60, "perception": 0.15},
    "diffusion": {"reach_short": 0.50, "grasp_angle": 0.20, "perception": 0.70},
}
SKILL = {"act": 0.85, "diffusion": 0.80}


def mock_rollout(policy, conditions, seed):
    """Stand-in for a real rollout. Replace this, keep everything else.

    Scene difficulty is a deterministic function of the initial condition, which
    is what real evaluation mostly looks like: a given cube pose is either
    inside the policy's competence or it is not. The residual coin flip is
    small and comes from the rig, not the scene.
    """
    # Not `hash(policy)`: Python salts string hashing per process, so the demo
    # would print different numbers on every run. Course figures have to repeat.
    tag = int(hashlib.sha256(policy.encode()).hexdigest()[:8], 16) % 1000
    rng = np.random.default_rng(seed * 7919 + tag)
    load = {
        "reach_short": abs(conditions["cube_y"] - 0.22) / 0.08,   # 0 at sweet spot
        "grasp_angle": abs(conditions["cube_yaw"]) / 0.79,
        "perception": abs(conditions["light"] - 1.0) / 0.4,
    }
    weighted = {tag: WEAKNESS[policy][tag] * value for tag, value in load.items()}
    difficulty = sum(weighted.values())
    success = difficulty < SKILL[policy] + rng.normal(0, 0.05)

    failure = None if success else max(weighted, key=weighted.get)
    return bool(success), failure, float(rng.uniform(6.0, 11.0))


DEMO = Protocol(
    task="cube to bin, 2026-08-09 rig",
    policies=("act", "diffusion"),
    n_trials=30,
    seed0=1000,
    factors={"cube_x": [-0.10, 0.10], "cube_y": [0.14, 0.30],
             "cube_yaw": [-0.79, 0.79], "light": [0.6, 1.4]},
    success_rule="cube fully inside the bin footprint at t=20s, gripper open",
    failure_tags=("reach_short", "grasp_angle", "perception", "timeout"),
    notes="mock rollouts; replace mock_rollout with the simulator",
)


# ------------------------------------------------------------------- coverage
def coverage_report(sims=200_000, seed=5):
    """Does a '95%' interval actually contain the truth 95% of the time?"""
    rng = np.random.default_rng(seed)
    print("\n  nominal 95% coverage, measured over "
          f"{sims:,} simulated evaluations")
    print(f"  {'n':>5}{'true rate':>12}{'Wald':>10}{'Wilson':>10}")
    for n in (10, 20, 50, 100):
        for p in (0.7, 0.9):
            k = rng.binomial(n, p, sims)
            p_hat = k / n
            half_w = Z95 * np.sqrt(p_hat * (1 - p_hat) / n)
            cover_w = np.mean((p_hat - half_w <= p) & (p <= p_hat + half_w))
            denom = 1 + Z95 ** 2 / n
            centre = (p_hat + Z95 ** 2 / (2 * n)) / denom
            half_s = Z95 / denom * np.sqrt(p_hat * (1 - p_hat) / n + Z95 ** 2 / (4 * n * n))
            cover_s = np.mean((centre - half_s <= p) & (p <= centre + half_s))
            print(f"  {n:>5}{p:>12.2f}{cover_w:>10.1%}{cover_s:>10.1%}")


def pairing_report(n=20, reps=1000, seed=99):
    """Does giving both policies the identical scenes actually buy anything?

    Same mock robot, same n, two ways of assigning scenes:
      paired   - policy A and policy B face the same n initial conditions
      unpaired - each policy gets its own n, drawn from the same distribution
    """
    rng = np.random.default_rng(seed)
    scenes = {}                                   # seed -> conditions, computed once
    paired, unpaired = [], []
    for _ in range(reps):
        base = int(rng.integers(0, 10 ** 6)) * 4

        def rate(policy, first_seed):
            hits = 0
            for i in range(n):
                s = first_seed + i
                if s not in scenes:
                    scenes[s] = sample_conditions(DEMO, s)
                hits += mock_rollout(policy, scenes[s], s)[0]
            return hits / n

        paired.append(rate("act", base) - rate("diffusion", base))
        unpaired.append(rate("act", base) - rate("diffusion", base + 2 * n))

    paired, unpaired = np.array(paired), np.array(unpaired)
    print(f"\n  n = {n} trials per policy, {reps} simulated evaluations")
    print(f"  {'assignment':<12}{'mean gap':>10}{'sd of gap':>12}{'ranks A above B':>18}")
    for name, d in (("unpaired", unpaired), ("paired", paired)):
        print(f"  {name:<12}{d.mean():>+10.3f}{d.std():>12.3f}{(d > 0).mean():>17.0%}")
    print(f"  pairing shrinks the spread by {unpaired.std() / paired.std():.2f}x")


# ----------------------------------------------------------------------- main
def main(out="runs/demo.jsonl"):
    run = EvalRun(DEMO, out)
    already = run.done()
    for policy, index, seed in paired_schedule(DEMO):
        if (policy, index) in already:
            continue
        conditions = sample_conditions(DEMO, seed)
        success, failure, seconds = mock_rollout(policy, conditions, seed)
        run.record(Trial(run.protocol_id, policy, index, seed,
                         conditions, success, failure, seconds))
    report(out)


if __name__ == "__main__":
    if "--coverage" in sys.argv:
        coverage_report()
        sys.exit()
    if "--pairing" in sys.argv:
        pairing_report()
        sys.exit()

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "gate.jsonl"

        # The protocol gate: an edited protocol cannot append to an old log.
        EvalRun(DEMO, probe)
        try:
            EvalRun(replace(DEMO, n_trials=DEMO.n_trials + 1), probe)
            raise AssertionError("protocol gate did not fire")
        except ValueError:
            pass

    sched = paired_schedule(DEMO)
    assert len(sched) == DEMO.n_trials * len(DEMO.policies)
    seeds = {i: {s for _, j, s in sched if j == i} for i in range(DEMO.n_trials)}
    assert all(len(s) == 1 for s in seeds.values()), "pairing broken"
    assert sched[0][0] != sched[2][0], "policy order does not rotate"
    print("protocol gate, pairing and rotation all check out.")

    # Real log on disk, so you can read it: head -3 runs/demo.jsonl
    main(Path(__file__).resolve().parent / "runs" / "demo.jsonl")
