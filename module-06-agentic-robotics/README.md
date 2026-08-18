# Module 6 - Agentic Robotics: the slow brain over learned skills

**Goal:** build the architecture the field is converging on, and be fluent on both sides of its seam. A slow deliberate layer - an LLM or VLM planner - directs a fast reactive layer, your trained policies, through a skill API that is honest about what happened. "Clear the desk" becomes perceive, plan, call a skill, watch it, judge it, recover. Most people are fluent on one side of that boundary. After this module you are fluent on both, and you can say why the boundary itself is the design.

**Two paths, and this module is the one where they nearly meet.** Nothing here needs a robot. Every exercise is standard-library Python and the milestone runs against a mock desk on purpose, because the failures this module is about are failures of interface and timing rather than of torque. Lessons 3 and 13 carry a short **Without hardware** section naming the two things you cannot measure - a precondition against a sensor you do not own, and the bus and camera latencies that are physical constants - and [lesson 14](https://robotics.biblio.guru/course/agentic-robotics/project-clear-the-desk/) states a required simulation criterion plus an optional hardware one held to the same standard. If you took the simulation path in [Module 4](https://robotics.biblio.guru/course/hardware/the-simulation-path/), you lose almost nothing here.

**Prerequisites:** Module 5 (you have a policy worth calling and an eval harness worth trusting) and Module 4 (you know what a bench does to a plan). Agent and tool-use experience helps and is not assumed; every protocol idea is defined at first use.

**Time:** about 6.6 hours of reading across thirteen lessons, plus a five-hour milestone; 3-4 weeks at course pace with the exercises. **Compute:** the lightest module since Module 1. **Hardware:** none required, and an optional extension if you have it.

## Lessons

### The architecture, and the number that shapes it

1. [The two-brain architecture](https://robotics.biblio.guru/course/agentic-robotics/the-two-brain-architecture/) - an agentic robot is three loops nested by rate, and the only interesting design question is what representation crosses the boundary between them.
2. [Why planners are slow](https://robotics.biblio.guru/course/agentic-robotics/why-planners-are-slow/) - a planner call costs a second or two for four reasons that add rather than trade, and that one number decides how you carve up every skill in the system.

### From a trained policy to a callable skill

3. [From policies to skills](https://robotics.biblio.guru/course/agentic-robotics/from-policies-to-skills/) - a trained policy has no beginning, no end and no opinion about whether it should be running; a skill is that policy plus a contract, and the contract is all of the work.
4. [Designing a skill API](https://robotics.biblio.guru/course/agentic-robotics/designing-a-skill-api/) - a contract offered to a model that cannot see your code, cannot feel your robot, and will take every name you choose literally.
5. [Preconditions, postconditions, and honest success](https://robotics.biblio.guru/course/agentic-robotics/preconditions-and-postconditions/) - a skill that reports success because the motion finished is the most expensive lie in the stack, because the planner believes it and never checks again.
6. [Grounding: what the robot actually sees](https://robotics.biblio.guru/course/agentic-robotics/grounding-what-the-robot-sees/) - turning the word "mug" into something an arm can reach for takes three separate steps, every one of them fails silently, and a confident wrong answer looks exactly like a correct one.

### On a wire, and around a loop

7. [Exposing skills over MCP: a tool that moves matter](https://robotics.biblio.guru/course/agentic-robotics/exposing-skills-over-mcp/) - the protocol is the part you already know; what is new is a tool that is slow, cannot be retried for free, fails physically, and can return success while its postcondition is false.
8. [The planner loop: perceive, plan, call, monitor, replan](https://robotics.biblio.guru/course/agentic-robotics/the-planner-loop/) - a plan is a hypothesis about a world you looked at once, and the loop's only job is keeping the model's picture of the desk matched to the desk.

### When it goes wrong

9. [Three ways a plan dies](https://robotics.biblio.guru/course/agentic-robotics/failure-taxonomy/) - a failed run has three possible authors, the skill, the plan, or the world model, and each one wants a different repair.
10. [Recovery is a choice, not a reflex](https://robotics.biblio.guru/course/agentic-robotics/monitoring-and-recovery/) - retry is one rung on a ladder of six, correct for exactly one failure class, and what picks the rung is a monitor that watches the world rather than the log.
11. [The refusal has to live below the model](https://robotics.biblio.guru/course/agentic-robotics/safety-and-the-refusal-boundary/) - a model's refusal is a property of text and a robot's harm is a joint trajectory, so the only layer that can really say no is the skill layer, in code the model cannot argue with.

### Judging it

12. [What a success rate means when the task has twenty steps](https://robotics.biblio.guru/course/agentic-robotics/evaluating-an-agentic-system/) - two honest numbers can describe the same ten runs and differ by more than fifty points, because partial credit and full-task success are not the same measurement.
13. [Latency budgets: three clocks that must never wait on each other](https://robotics.biblio.guru/course/agentic-robotics/latency-budgets/) - the whole design problem is keeping each tier busy while the tier above it thinks.

### Milestone

14. [Clear the desk, and the three failures that need three different fixes](https://robotics.biblio.guru/course/agentic-robotics/project-clear-the-desk/) - the full agentic loop against a mock desk with three failure dials, then a measurement of which recovery mechanism buys back which failure class.

Figures live in [`assets/`](assets/) as hand-authored SVG, so they diff and review like any other source. The milestone scaffold is in [`project/`](project/): [`clear_the_desk.py`](project/clear_the_desk.py) ships a complete world, skill layer and scorer, and stubs exactly the three pieces that make it an agent - the planner loop, the goal check and the episode driver. There is deliberately no solutions directory. The interesting part of this milestone is the ablation you run on your own loop, and a reference loop would hand you the answer the experiment exists to produce.

## Milestone project (summary; full spec in [lesson 14](https://robotics.biblio.guru/course/agentic-robotics/project-clear-the-desk/))

**Required on both paths.** A planner over a skill library, clearing a simulated desk of four filable objects plus one that belongs nowhere and must be left alone. Three dials inject three failure classes independently: a grasp that misses, a detector that omits or mislabels an object, and a world that drifts under a plan made a moment ago. Build it in five checkpoints, then cross three loops - plan once, retry on error, re-perceive every cycle - with three dial settings, five hundred episodes per cell. Report full-task success and step completion side by side for all nine. The finding is the cell where a recovery mechanism does nothing at all, because that is the failure class your loop cannot see from the inside.

**If you have the arm.** Point the same agent at your own MCP server and policies and clear a real desk ten times, uncut, from a written list of layouts fixed beforehand, with the goal check reading the desk rather than the transcript, causes drawn from the same fixed list plus an *unanticipated* bucket written out individually, and every intervention counted. Ten runs cannot separate anything from anything, so report them as a failure catalogue against the mock table rather than as a rate. If you then fit the dials to reproduce what you saw, that is a second result and it is printed second.

## Exit test

Draw the three tiers cold, with rates, deadlines and what crosses each boundary. Take a failed run and attribute it to the skill, the plan or the world model, naming the evidence that settles it. Say why verification belongs between skills rather than inside a policy success flag, and why the refusal gate cannot read the justification attached to a call. Give an end-to-end latency budget for your own stack and name the mechanism hiding each delay. Finally, say what changes if the planner becomes a VLA with its own deliberate layer, and at what point this whole tier dissolves into the model.
