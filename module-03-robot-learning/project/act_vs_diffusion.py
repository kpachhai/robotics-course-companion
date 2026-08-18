"""Module 3 milestone - ACT against Diffusion Policy, on two tasks, at two data budgets.

This is a scaffold, not a starter-with-blanks. The experiment is yours to build;
what is written here is the bookkeeping that makes the result defensible, plus a
`--check` gate that decides whether you are finished. There is no solution file
for this one, deliberately.

    python act_vs_diffusion.py --plan       list every run the matrix asks for
    python act_vs_diffusion.py --train      train the cells that are missing
    python act_vs_diffusion.py --eval       run the trials that are missing
    python act_vs_diffusion.py --report     the scoreboard, computed from the log
    python act_vs_diffusion.py --scaling    the 10-versus-50 table
    python act_vs_diffusion.py --check      does this count as done? mechanically
    python act_vs_diffusion.py --writeup    emit the writeup skeleton, numbers filled

Two paths through the same protocol. Pick one with --path and never mix them in
one log:

    laptop   the module's own chunked policies on the two sim tasks, CPU only.
             Task A is `code/groove_world.py`; task B is the corridor from
             `code/compare_policies.py`. Everything runs here, slowly but freely.

    gpu      real LeRobot on a rented box or a Colab notebook. Task A is your
             Module 2 recording, task B is `lerobot/pusht`. `train_policy` shells
             out to `lerobot-train`, `run_trial` to the rollout entry point of
             whichever LeRobot version you installed. Check the CLI names against
             your own install: they changed in v0.6.0 (2026-07-06).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))       # your completed lesson code

# Your lesson 3.13 harness. Everything below leans on it rather than
# reimplementing logs, protocols, intervals or the paired comparison.
from eval_harness import (EvalRun, Protocol, Trial, compare,  # noqa: E402
                          sample_conditions, summarise, wilson_interval)

# --------------------------------------------------------------------------
# 1. THE PROTOCOL. Fill this in before the first trial exists.
#
# The two blanks below are not decoration. A success predicate written after you
# have seen a few rollouts is a predicate fitted to your policy, and it is the
# cheapest way in this whole project to publish a number that means nothing.
# The file refuses to run until both are filled.

SUCCESS_RULE = ""            # TODO(you): one sentence. What counts as success?
FAILURE_TAGS: tuple[str, ...] = ()   # TODO(you): the failure modes you will tag

TASKS = ("task_a", "task_b")
POLICIES = ("act", "diffusion")
DEMO_BUDGETS = (10, 50)
SEEDS = (0, 1, 2)
N_TRIALS = 50                # per cell. In sim, trials are cheap. Spend them.

# One ablation per policy, run on task A at 50 demonstrations only.
#   act:       the chunk size, the knob lesson 3.6 measured
#   diffusion: the observation space, the knob lesson 3.2 argued about
ABLATIONS = {
    "act": {"chunk_size": None},          # TODO(you): the value you are testing
    "diffusion": {"n_obs_steps": None},   # TODO(you): the value you are testing
}


@dataclass(frozen=True)
class Cell:
    """One trained policy. The unit the matrix is made of."""
    task: str
    policy: str
    demos: int
    seed: int
    ablation: str = ""       # "" for the baseline cells

    @property
    def arm(self) -> str:
        """The thing being compared: a policy, possibly with one knob moved."""
        return f"{self.policy}@{self.ablation}" if self.ablation else self.policy

    @property
    def key(self) -> str:
        """One trained policy, inside a log. Seeds stay separate here on purpose."""
        return f"{self.arm}.s{self.seed}"

    @property
    def name(self) -> str:
        return f"{self.task}.d{self.demos}.{self.key}"


def matrix() -> list[Cell]:
    """Every run this project asks for. 24 in the grid, 6 in the ablations."""
    cells = [Cell(task, policy, demos, seed)
             for task in TASKS for policy in POLICIES
             for demos in DEMO_BUDGETS for seed in SEEDS]
    cells += [Cell("task_a", policy, 50, seed, ablation=next(iter(knobs)))
              for policy, knobs in ABLATIONS.items() for seed in SEEDS]
    return cells


def protocol_for(task: str, policies: tuple[str, ...]) -> Protocol:
    """One frozen protocol per task. Its fingerprint stamps every trial."""
    if not SUCCESS_RULE or not FAILURE_TAGS:
        raise SystemExit(
            "Write SUCCESS_RULE and FAILURE_TAGS before running anything.\n"
            "Deciding what counts as success after you have watched a rollout is "
            "how an evaluation becomes a description of the policy you happen to have."
        )
    return Protocol(
        task=task,
        policies=policies,
        n_trials=N_TRIALS,
        seed0=10_000,
        factors=FACTORS[task],
        success_rule=SUCCESS_RULE,
        failure_tags=FAILURE_TAGS,
        notes=f"module 3 milestone, {N_TRIALS} trials per cell, seeds {SEEDS}",
    )


# TODO(you): the initial conditions you will vary, per task, as name -> [low, high].
# Everything you do NOT list here is frozen, and the frozen list is the more
# informative half of your claim (lesson 3.12).
FACTORS: dict[str, dict] = {
    "task_a": {},
    "task_b": {},
}


# --------------------------------------------------------------------------
# 2. THE TWO ADAPTERS. This is the whole exercise.
#
# Everything else in this file is task-agnostic and policy-agnostic on purpose:
# if you find yourself editing the runner to make a policy fit, the interface is
# wrong, and it will be wrong again in Module 5 when you swap in a VLA.

def train_policy(cell: Cell):
    """Train one cell and return a handle `run_trial` can use.

    A handle is whatever your path needs: an in-memory model on the laptop path,
    a checkpoint directory on the gpu path. Two rules that are not negotiable.

    Same budget for both policies. Matched gradient steps, matched batch size,
    matched data. An unmatched comparison measures your patience, not the method.

    Same demonstrations within a (task, demos, seed) triple. Both policies must
    see the identical episodes, or the data budget stops being a controlled
    variable. Derive the demonstration set from `cell.seed`, never from a global.
    """
    raise NotImplementedError("TODO(you): train one cell")


def run_trial(handle, cell: Cell, conditions: dict, seed: int):
    """Run one rollout of a trained policy from one initial condition.

    Returns (success: bool, failure: str | None, seconds: float). `failure` must
    be one of FAILURE_TAGS whenever success is False; the harness rejects a tag
    it has not been told about, which is the point.

    Do not let this function decide what success means. It applies SUCCESS_RULE.
    """
    raise NotImplementedError("TODO(you): roll out one trial")


# --------------------------------------------------------------------------
# 3. THE RUNNER. Given complete. Resumable, paired, interleaved.

def log_path(task: str, demos: int) -> Path:
    return HERE / "runs" / f"{task}-d{demos}.jsonl"


def group(cells: list[Cell]) -> dict:
    """Cells that share a log, and therefore share scenes: one task, one budget.

    The ablation arms live in the SAME log as the baseline they are an ablation
    of. That is the whole reason the ablation is worth running: it is a paired
    comparison against the unmodified policy on identical scenes, and putting it
    in its own file would throw the pairing away.
    """
    out: dict = {}
    for cell in cells:
        out.setdefault((cell.task, cell.demos), []).append(cell)
    return out


def evaluate_all(path: str, cells: list[Cell]) -> None:
    """Train what is missing, roll out what is missing, append as you go.

    The (key, index) resume key comes from the harness, so an interrupted
    session costs you the trial you were in the middle of and nothing else.
    """
    for (task, demos), members in group(cells).items():
        keys = tuple(sorted(c.key for c in members))
        proto = protocol_for(task, keys)
        run = EvalRun(proto, log_path(task, demos))
        already = run.done()
        handles = {}

        for key, index, seed in _schedule(proto, keys):
            if (key, index) in already:
                continue
            cell = next(c for c in members if c.key == key)
            if key not in handles:
                print(f"  training {cell.name} ({path} path)")
                handles[key] = train_policy(cell)
            conditions = sample_conditions(proto, seed)
            success, failure, seconds = run_trial(handles[key], cell, conditions, seed)
            run.record(Trial(run.protocol_id, key, index, seed, conditions,
                             bool(success), failure, float(seconds)))


def _schedule(proto: Protocol, keys: tuple[str, ...]):
    """Paired and interleaved, over every cell rather than over two policies.

    Lesson 3.13 paired two policies. Here a cell is a (policy, ablation, seed)
    triple, and every one of them must face the identical scene at a given trial
    index, or the seed spread and the trial spread get tangled together.
    """
    order = list(keys)
    for i in range(proto.n_trials):
        shift = i % len(order)
        for key in order[shift:] + order[:shift]:
            yield key, i, proto.seed0 + i


# --------------------------------------------------------------------------
# 4. THE REPORT. Given complete: numbers come from the log or not at all.

def load_all() -> dict:
    out = {}
    for path in sorted((HERE / "runs").glob("*.jsonl")):
        header, trials = EvalRun.load(path)
        out[path.stem] = (header, trials)
    return out


def report() -> None:
    for name, (header, trials) in load_all().items():
        print(f"\n{name}   protocol {header['protocol_id']}   n = {N_TRIALS} per cell")
        print(f"  {'cell':<20}{'rate':>10}   95% Wilson        top failure")
        for key, s in summarise(trials).items():
            worst = s["failures"].most_common(1)
            tag = f"{worst[0][0]} x{worst[0][1]}" if worst else "-"
            print(f"  {key:<20}{s['k']:>4}/{s['n']:<4}{s['rate']:>5.0%}   "
                  f"[{s['lo']:.2f}, {s['hi']:.2f}]   {tag}")

        pooled = _pool_seeds(trials)
        print(f"\n  seeds pooled, {len(_arms(trials))} arms:")
        for arm, (k, n, spread) in _arm_rates(trials).items():
            lo, hi = wilson_interval(k, n)
            print(f"  {arm:<20}{k:>4}/{n:<4}{k / n:>5.0%}   [{lo:.2f}, {hi:.2f}]   "
                  f"seed spread {spread:.0%}")
        for a, b in _pairs_to_test(trials):
            c = compare(pooled, a, b)
            print(f"  {a} vs {b}: {c['a_only']} / {c['b_only']} disagreements "
                  f"over {c['n_pairs']} scenes, McNemar p = {c['p_value']:.3f}")


def _arms(trials) -> set:
    return {t.policy.rsplit(".s", 1)[0] for t in trials}


def _pool_seeds(trials):
    """`act.s0` becomes `act`, so the paired test sees arms rather than cells."""
    return [type(t)(**{**asdict(t), "policy": t.policy.rsplit(".s", 1)[0]})
            for t in trials]


def _arm_rates(trials) -> dict:
    """Pooled rate per arm, plus the seed-to-seed spread it is hiding."""
    out = {}
    for arm in sorted(_arms(trials)):
        mine = [t for t in trials if t.policy.startswith(arm + ".s")]
        by_seed = {}
        for t in mine:
            by_seed.setdefault(t.policy, []).append(t.success)
        rates = [sum(v) / len(v) for v in by_seed.values()]
        out[arm] = (sum(t.success for t in mine), len(mine), max(rates) - min(rates))
    return out


def _pairs_to_test(trials):
    """act against diffusion, and each ablation arm against its own baseline."""
    arms = _arms(trials)
    pairs = []
    if {"act", "diffusion"} <= arms:
        pairs.append(("act", "diffusion"))
    pairs += [(arm.split("@")[0], arm) for arm in sorted(arms) if "@" in arm]
    return pairs


def scaling() -> None:
    """The 10-versus-50 table. The real lesson of the project lives here."""
    rows = {}
    for name, (_, trials) in load_all().items():
        task, demos = name.rsplit("-d", 1)
        for arm, (k, n, _) in _arm_rates(trials).items():
            if "@" not in arm:                       # ablation arms sit this out
                rows[(task, arm, int(demos))] = (k, n)

    print(f"\n{'task':<10}{'policy':<12}{'10 demos':>20}{'50 demos':>20}{'gap':>8}")
    for task in TASKS:
        for policy in POLICIES:
            cells = [rows.get((task, policy, d)) for d in DEMO_BUDGETS]
            if not all(cells):
                continue
            text = []
            for k, n in cells:
                lo, hi = wilson_interval(k, n)
                text.append(f"{k / n:.0%} [{lo:.2f},{hi:.2f}]")
            gap = cells[1][0] / cells[1][1] - cells[0][0] / cells[0][1]
            print(f"{task:<10}{policy:<12}{text[0]:>20}{text[1]:>20}{gap:>+8.0%}")


# --------------------------------------------------------------------------
# 5. DONE MEANS DONE. Given complete, and not negotiable by editing prose.

WRITEUP = HERE / "writeup.md"
REQUIRED_SECTIONS = (
    "## What I ran",
    "## Results",
    "## The data-scaling curve",
    "## The ablations",
    "## What I could not separate",
    "## Failure modes",
    "## What I would do next",
)


def check() -> int:
    """Every criterion is a fact about files on disk. Nothing here is a judgement."""
    logs = load_all()
    failures = []

    def want(condition, message):
        print(f"  {'pass' if condition else 'FAIL'}  {message}")
        if not condition:
            failures.append(message)

    expected = group(matrix())
    want(len(logs) == len(expected),
         f"one log per (task, budget): {len(logs)} of {len(expected)}")

    for name, (header, trials) in logs.items():
        task, demos = name.rsplit("-d", 1)
        wanted = {c.key for c in expected.get((task, int(demos)), [])}
        keys = {t.policy for t in trials}
        counts = {k: sum(1 for t in trials if t.policy == k) for k in keys}
        want(keys == wanted,
             f"{name}: {len(keys)} cells logged, expected {len(wanted)}")
        want(all(n >= N_TRIALS for n in counts.values()),
             f"{name}: every cell has at least {N_TRIALS} trials")
        want(all(t.protocol_id == header["protocol_id"] for t in trials),
             f"{name}: one protocol id across the whole file")
        want(all(t.failure for t in trials if not t.success),
             f"{name}: every failure carries a tag")
        by_index: dict = {}
        for t in trials:
            by_index.setdefault(t.index, set()).add(t.policy)
        want(all(len(v) == len(keys) for v in by_index.values()),
             f"{name}: every scene was run by every cell (pairing intact)")

    text = WRITEUP.read_text() if WRITEUP.exists() else ""
    want(bool(text), f"{WRITEUP.name} exists")
    for section in REQUIRED_SECTIONS:
        want(section in text, f"writeup has the {section!r} section")
    want("n =" in text or "n=" in text, "writeup states a trial count")
    want("[" in text and "," in text, "writeup reports intervals, not bare rates")

    print(f"\n{'DONE' if not failures else str(len(failures)) + ' criteria unmet'}")
    return 0 if not failures else 1


def writeup_skeleton() -> None:
    """Print the writeup with the numbers already filled in. The prose is yours."""
    print(f"# ACT against Diffusion Policy, on two tasks\n")
    print(f"## What I ran\n\n<!-- paths, budgets, hardware, wall-clock, and the "
          f"exact protocol fingerprints below -->\n")
    for name, (header, _) in load_all().items():
        print(f"- `{name}`: protocol `{header['protocol_id']}`, "
              f"success rule: {header['protocol']['success_rule']}")
    print("\n## Results\n")
    report()
    print("\n## The data-scaling curve\n")
    scaling()
    print("\n## The ablations\n\n<!-- what changed, and whether it changed anything "
          "you can defend -->\n")
    print("## What I could not separate\n\n<!-- the pairs whose intervals overlap. "
          "Name them. This section is the one that makes the rest believable -->\n")
    print("## Failure modes\n\n<!-- the tag histogram, and what each tag would take "
          "to fix -->\n")
    print("## What I would do next\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", choices=("laptop", "gpu"), default="laptop")
    for flag in ("plan", "train", "eval", "report", "scaling", "check", "writeup"):
        ap.add_argument(f"--{flag}", action="store_true")
    args = ap.parse_args()

    if args.plan:
        cells = matrix()
        for cell in cells:
            print(f"  {cell.name}")
        print(f"\n{len(cells)} training runs, {len(cells) * N_TRIALS:,} trials, "
              f"{len(group(cells))} logs")
        return 0
    if args.train or args.eval:
        evaluate_all(args.path, matrix())
        return 0
    if args.report:
        report()
        return 0
    if args.scaling:
        scaling()
        return 0
    if args.writeup:
        writeup_skeleton()
        return 0
    if args.check:
        return check()
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
