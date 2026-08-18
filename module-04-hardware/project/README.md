# Milestone: a robot doing a real task

The specification. The lesson explains the reasoning; this file is the checklist you
work against and the thing you check yourself off on.

There is no solution file for this milestone, deliberately. The whole module leads here.

**Two criteria.** This file states the hardware one in full, because it is the one with
irreversible steps in it. If you took the simulation path from
[lesson 0](https://robotics.biblio.guru/course/hardware/the-simulation-path/), read the *Simulation path* note under each heading
below, and the full column in [lesson 18](https://robotics.biblio.guru/course/hardware/project-a-real-robot-doing-a-real-task/).
Neither criterion is the easy one: the hardware reader is limited by wall-clock and pays for it
in a wide interval, and the simulation reader is limited by nothing and owes a narrow one.

## The claim you are making

> A policy I trained does *this task* at *this success rate*, over *this many uncut
> consecutive trials*, with the object starting anywhere in *this stated distribution*.

Everything below exists to make that sentence checkable by somebody who was not there.

## Target

- **At least 70% success over at least 20 consecutive uncut trials**, in one sitting.
- Twenty is the floor, not the goal. At 70% observed, twenty trials give a 95% interval
  of roughly 48-85%; thirty give roughly 52-83%; forty give roughly 55-82%. If you can
  run more, run more, and say how many you ran.

*Simulation path:* **200 scored trials per policy**, initial conditions dealt from a seed
never used in training, and the bar is 70% with the **lower bound of the Wilson interval above
60%**. At 200 trials a 70% rate carries an interval of roughly 63-76%. Twenty trials would be a
choice here rather than a constraint, and it would leave every interesting comparison inside the
interval.

## Task selection

Pick a task in the middle band:

- Not "push a block", which succeeds without a policy and teaches you nothing.
- Not "fold a cloth" or "insert a plug", which are contact-rich and need hundreds of
  episodes on hardware that cannot command force.
- Something like: a cube into a bowl from four start zones, or a two-colour sort.

**The entry test:** you can teleoperate it ten times out of ten, smoothly, with the same
grasp and approach every time. If you cannot, no dataset of it will be consistent, and
the policy will learn your inconsistency faithfully.

## Checkpoints

Each one has a condition. Moving on is not the same as passing.

| # | Checkpoint | Done when |
|---|---|---|
| 1 | Freeze the rig | Cameras clamped, table marked, lighting artificial and controlled; you teleoperate the task 10/10 |
| 2 | Record ~50 episodes | Dealt from a coverage plan, viewed in the dataset viewer, bad episodes discarded at record time |
| 3 | Train | Loss fell and flattened; five to ten epochs, not an arbitrary step count |
| 4 | First rollout | Short duration, per-step motion clamped, safety glasses on, power plug within reach |
| 5 | Twenty trials | One sitting, camera running throughout, protocol written *before* trial one |

Expect to return to checkpoint 2 at least twice. That return trip is the project.

*Simulation path:* the same five checkpoints with the rig freeze becoming a scene-file hash and
a pinned simulator version, checkpoint 2 becoming **two** datasets of fifty episodes on the same
schedule (one teleoperated by hand, one from the scripted controller), checkpoint 4 losing its
safety clauses, and checkpoint 5 becoming 200 trials per policy. Then two more:

| # | Checkpoint | Done when |
|---|---|---|
| 6 | The demonstration-quality experiment | Both policies scored under one protocol, both rates with intervals, and a sentence on how much of the gap was consistency rather than capacity |
| 7 | The robustness table | The same 200-trial protocol re-run under four perturbations never trained on, one at a time: object mass x1.5, friction x0.7, one camera moved 2 cm, light direction rotated |

## Deliverables

- [ ] `protocol.md` committed **before** the first trial: initial-condition distribution,
      success criterion in one sentence, planned trial count, what makes a trial void
- [ ] `results.jsonl` from `run_eval.py`, one line per trial, written as you went
- [ ] The report from `eval_report.py`, including the per-condition breakdown
- [ ] Dataset pushed to the Hub, episode count stated
- [ ] Checkpoint pushed, with a model card naming the LeRobot version, policy type,
      training steps, camera models and positions, and lighting
- [ ] One photograph of the rig
- [ ] Uncut video of all trials, in order, real time (or labelled speed)
- [ ] Failure reel: every failure, with its tag
- [ ] Writeup: what you tried, what the first policy did, what you changed, what happened
      next. Lead with the interval, not the point estimate.

*Simulation path:* the same list, with the rig photograph replaced by the scene file plus its
content hash and the simulator version, the uncut video replaced by an uncut screen recording of
the scored trials in dealt order plus the seed that reproduces them, and three additions: the
hand-versus-scripted comparison table, the four-row robustness table, and one paragraph reporting
chunk-level action error from training the same architecture on a pinned public SO-101 dataset,
explaining why that number is not a success rate. The headline sentence names the venue.

## The scripts

`run_eval.py` - the trial harness. Deals initial conditions in a fixed random order,
prompts you per trial, tags failures, appends one JSON line as you go. Four things are
left for you to fill in, because they are the experiment: the task name, the success
criterion, the failure buckets, and the initial-condition distribution. The script
refuses to run until they are filled in.

`eval_report.py` - reads the log and prints the rate, its interval, the void count, the
failure histogram, a per-condition breakdown, and a markdown block for the writeup.

```
python run_eval.py --dry-run --trials 20        # check the conditions look right
python run_eval.py --trials 20 --out results.jsonl --notes "act-40k, lamp on"
python eval_report.py results.jsonl --group cell
```

Neither script talks to the robot. You run the rollout in another terminal and type the
verdict here, on purpose: the thing that scores the experiment should not be the thing
running it.

## Safety, before the first autonomous rollout

Hardware path only, and nothing on this list has a simulated analogue. Read it anyway: the first
time you meet a powered arm should not also be the first time you read that cutting power drops
it rather than freezing it.

The arm does not know where your hands are. There is no collision detection, no force
sensing and no e-stop.

- Safety glasses.
- The follower's power plug loose and within reach of your non-dominant hand.
- A clear, padded swept volume; nothing fragile, no open drinks near the board.
- First rollout short, with per-step motion clamped.
- Remember that cutting power does not freeze the arm, it drops it. Plan for the fall.
