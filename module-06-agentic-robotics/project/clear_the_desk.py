"""Module 6 milestone - clear the desk (Lesson 6.14).

A scaffold, not a starter-with-blanks. The world, the skill layer and the
scoring helpers below are complete and deliberately unforgiving. The three
things that make it an agent - the planner loop, the goal check and the
episode driver - are stubbed with NotImplementedError. There is no solution
file for this one, on purpose.

Standard library only, so it runs anywhere and 200 episodes cost seconds.
When your loop works here, swap SkillServer for the MCP server from lesson
6.7 without touching the agent: the result shape is the same.

Run:  python clear_the_desk.py --demo
      python clear_the_desk.py -n 200 --seed 1
      python clear_the_desk.py -n 200 --grasp-p 0.7 --miss-p 0.1 --drift-p 0.1
"""
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field

# ---------------------------------------------------------------- the domain

BIN_FOR_KIND = {"mug": "sink", "pen": "tray", "book": "shelf"}
BINS = ("sink", "tray", "shelf")
KINDS = tuple(BIN_FOR_KIND) + ("gadget",)   # gadget has no bin, on purpose

# Nominal wall-clock cost of each skill, for the latency budget in lesson 6.13.
SKILL_SECONDS = {"look": 0.8, "pick": 4.0, "place": 3.5, "home": 2.0}


@dataclass
class Item:
    item_id: str
    kind: str
    x: float
    y: float
    in_bin: str | None = None


@dataclass
class Desk:
    """Ground truth. Only the skill layer and your evaluator may read this.
    Your agent must never touch it directly - that is the whole experiment."""
    items: list[Item]
    holding: str | None = None
    robot_seconds: float = 0.0
    drift_events: list[str] = field(default_factory=list)

    def on_desk(self) -> list[Item]:
        return [i for i in self.items
                if i.in_bin is None and i.item_id != self.holding]


def make_desk(rng: random.Random, n_items: int = 5) -> Desk:
    kinds = [rng.choice(("mug", "pen", "book")) for _ in range(n_items - 1)]
    kinds.append("gadget")               # every desk has one thing you cannot file
    rng.shuffle(kinds)
    items = [Item(f"obj{n}", k, round(rng.uniform(-0.3, 0.3), 3),
                  round(rng.uniform(0.15, 0.45), 3))
             for n, k in enumerate(kinds)]
    return Desk(items=items)


# ------------------------------------------------------------- the skill API
# The shape a planner sees. Descriptions are load-bearing: they are the only
# preconditions the protocol will carry for you (lesson 6.5).

TOOLS = [
    {"name": "look",
     "description": "Report every object currently on the desk. Detection is "
                    "imperfect: an object may be missed or mislabelled, and a "
                    "confident label is not evidence.",
     "behavior": "BLOCKING",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "pick",
     "description": "Close the gripper on one object. Requires an empty gripper "
                    "and an object with this id on the desk. On failure the "
                    "gripper is empty and the object may have moved.",
     "behavior": "BLOCKING",
     "inputSchema": {"type": "object",
                     "properties": {"item_id": {"type": "string"}},
                     "required": ["item_id"]}},
    {"name": "place",
     "description": f"Drop the held object into a bin. Bins are {list(BINS)}. "
                    "Requires a held object.",
     "behavior": "BLOCKING",
     "inputSchema": {"type": "object",
                     "properties": {"bin": {"type": "string", "enum": list(BINS)}},
                     "required": ["bin"]}},
    {"name": "home",
     "description": "Return the arm to its rest pose. Always safe.",
     "behavior": "BLOCKING",
     "inputSchema": {"type": "object", "properties": {}}},
]


def _ok(text: str, data: dict | None = None) -> dict:
    return {"content": [{"type": "text", "text": text}],
            "structuredContent": data or {}, "isError": False}


def _err(text: str) -> dict:
    """A tool execution error: a successful call reporting a failed action.
    The planner is expected to read this and recover (lesson 6.9)."""
    return {"content": [{"type": "text", "text": text}],
            "structuredContent": {}, "isError": True}


class SkillServer:
    """Four skills over one desk, with the three failure sources you will
    measure: unreliable grasping, imperfect detection, and a world that
    changes without telling anyone."""

    def __init__(self, desk: Desk, rng: random.Random, grasp_p: float = 0.85,
                 miss_p: float = 0.0, drift_p: float = 0.0):
        self.desk, self.rng = desk, rng
        self.grasp_p, self.miss_p, self.drift_p = grasp_p, miss_p, drift_p
        self.calls: list[tuple[str, dict, bool]] = []
        self._drifts_left = 1

    def list_tools(self) -> list[dict]:
        return TOOLS

    def call(self, name: str, **args) -> dict:
        handler = getattr(self, f"_{name}", None)
        if handler is None:                       # protocol error, not a tool error
            raise KeyError(f"no such skill: {name!r}")
        self.desk.robot_seconds += SKILL_SECONDS[name]
        result = handler(**args)
        self.calls.append((name, args, result["isError"]))
        self._maybe_drift()
        return result

    # -- skills ------------------------------------------------------------

    def _look(self) -> dict:
        seen = []
        for item in self.desk.on_desk():
            if self.rng.random() < self.miss_p:
                continue                                    # a silent omission
            kind = item.kind
            if self.rng.random() < self.miss_p:             # a confident mislabel
                kind = self.rng.choice([k for k in KINDS if k != item.kind])
            seen.append({"id": item.item_id, "kind": kind, "x": item.x, "y": item.y})
        return _ok(f"{len(seen)} object(s) on the desk", {"objects": seen})

    def _pick(self, item_id: str) -> dict:
        if self.desk.holding is not None:
            return _err(f"gripper is already holding {self.desk.holding}")
        match = [i for i in self.desk.on_desk() if i.item_id == item_id]
        if not match:
            return _err(f"no object with id {item_id!r} is on the desk")
        if self.rng.random() >= self.grasp_p:
            item = match[0]
            item.x = round(item.x + self.rng.uniform(-0.08, 0.08), 3)
            item.y = round(item.y + self.rng.uniform(-0.08, 0.08), 3)
            return _err(f"grasp of {item_id} failed: the gripper is empty and the "
                        "object may have moved. Perceive again before retrying.")
        self.desk.holding = item_id
        return _ok(f"holding {item_id}", {"holding": item_id})

    def _place(self, bin: str) -> dict:
        if self.desk.holding is None:
            return _err("gripper is empty; nothing to place")
        if bin not in BINS:
            return _err(f"no bin named {bin!r}; bins are {list(BINS)}")
        held = next(i for i in self.desk.items if i.item_id == self.desk.holding)
        held.in_bin, self.desk.holding = bin, None
        return _ok(f"placed {held.item_id} in {bin}", {"bin": bin})

    def _home(self) -> dict:
        return _ok("arm at rest")

    # -- the world moving on its own ---------------------------------------

    def _maybe_drift(self) -> None:
        """Someone sets another object down while the robot is working. Nothing
        reports this. A planner that looked once will never know."""
        if self._drifts_left <= 0 or self.rng.random() >= self.drift_p:
            return
        self._drifts_left -= 1
        kind = self.rng.choice(("mug", "pen", "book"))
        new = Item(f"obj{len(self.desk.items)}", kind,
                   round(self.rng.uniform(-0.3, 0.3), 3),
                   round(self.rng.uniform(0.15, 0.45), 3))
        self.desk.items.append(new)
        self.desk.drift_events.append(f"{new.item_id} ({kind}) appeared on the desk")


# ------------------------------------------------------------- yours to build

CAUSES = ("skill_failed", "plan_wrong", "world_changed", "grounding",
          "memory", "timeout")


def goal_check(desk: Desk) -> tuple[bool, list[str]]:
    """Grade the run from outside the agent, against ground truth.

    Returns (passed, unmet) where unmet names each subgoal that did not hold.
    Decide and write down what "clear" means before you write the planner:
    where the gadget belongs, whether the gripper must end empty, whether the
    arm must be home. TODO(you).
    """
    raise NotImplementedError


class Agent:
    """Your planner loop (lesson 6.8). It may only reach the world through
    `server.call(...)`, and it may only learn about the world from the results
    that come back. Charge self.thinking_seconds for every decision you make."""

    def __init__(self, server: SkillServer, planner_seconds: float = 1.2):
        self.server = server
        self.planner_seconds = planner_seconds
        self.thinking_seconds = 0.0
        self.interventions = 0

    def run(self, max_calls: int = 40) -> None:
        """Perceive, plan, call skills, read isError, recover, and stop when you
        believe the desk is clear or you run out of calls. TODO(you)."""
        raise NotImplementedError


def run_episode(rng: random.Random, grasp_p: float, miss_p: float, drift_p: float,
                planner_seconds: float, verbose: bool = False) -> dict:
    """One attempt. Build a desk, run your agent, then grade the desk - never
    the transcript - and attribute any failure to exactly one cause in CAUSES.

    Return dict(success, subgoals_met, subgoals_total, cause, interventions,
                robot_seconds, thinking_seconds, calls). TODO(you).
    """
    raise NotImplementedError


# ----------------------------------------------------------------- the report

def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a rate. Ten runs buy you less than you think (6.12)."""
    if n == 0:
        return (0.0, 1.0)
    p, denom = successes / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def report(results: list[dict]) -> None:
    n = len(results)
    wins = sum(r["success"] for r in results)
    lo, hi = wilson(wins, n)
    met = sum(r["subgoals_met"] for r in results)
    total = sum(r["subgoals_total"] for r in results) or 1
    think = sum(r["thinking_seconds"] for r in results)
    move = sum(r["robot_seconds"] for r in results)
    print(f"episodes            {n}")
    print(f"full-task success   {wins}/{n} = {wins / n:.1%}  "
          f"(95% CI {lo:.1%} to {hi:.1%})")
    print(f"step completion     {met / total:.1%}")
    print(f"interventions/task  {sum(r['interventions'] for r in results) / n:.2f}")
    print(f"idle fraction       {think / (think + move):.1%}  "
          f"({think:.0f} s thinking, {move:.0f} s moving)")
    print("failure causes")
    for cause in CAUSES:
        k = sum(1 for r in results if not r["success"] and r["cause"] == cause)
        if k:
            print(f"  {cause:<14} {k:>4}  ({k / max(1, n - wins):.0%} of failures)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Module 6 milestone: clear the desk")
    ap.add_argument("--demo", action="store_true", help="one verbose episode")
    ap.add_argument("-n", type=int, default=50, help="episodes to evaluate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grasp-p", type=float, default=0.85, help="P(pick succeeds)")
    ap.add_argument("--miss-p", type=float, default=0.0, help="P(look misses/mislabels)")
    ap.add_argument("--drift-p", type=float, default=0.0, help="P(world changes per call)")
    ap.add_argument("--planner-latency", type=float, default=1.2,
                    help="seconds charged per planner decision")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    episodes = 1 if args.demo else args.n
    results = [run_episode(rng, args.grasp_p, args.miss_p, args.drift_p,
                           args.planner_latency, verbose=args.demo)
               for _ in range(episodes)]
    report(results)


if __name__ == "__main__":
    main()
