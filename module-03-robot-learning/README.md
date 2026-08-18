# Module 3 - Robot Learning: policies from data

**Goal:** by the end of this module you can take a pile of demonstrations and turn it into a policy, know exactly why that policy falls apart the first time it acts, fix it three different ways, build the two architectures the field actually ships, and say whether any of it worked in a sentence that survives someone else re-running it. The through-line is that last part. Almost everything here is easy to make *look* like it works, and the difference between a demo and a result is a protocol you froze before the first trial.

**Prerequisites:** Module 2. You need the two-link arm and the pure-pursuit tracker from Module 1, because Lessons 3.4 and 3.18 clone them and then break them; you need your Module 2 recording, because most exercises take a `--path` pointing at it; and you need to have watched a scripted expert succeed, because the whole first half of this module is about why copying one is harder than it looks.

**Time:** ~10 hours of reading across the eighteen teaching lessons, plus about five hours on the milestone; 6-8 weeks at course pace with the exercises. **New dependency:** `torch`, first needed in Lesson 3.3, plus `pandas`, `pyarrow` and `huggingface_hub` for the dataset audit in 3.1. `mujoco` from Module 2 is reused once, in 3.16. **Hardware:** none required. Every exercise up to the milestone runs on a laptop CPU in a minute or less, and the two things a laptop genuinely cannot do - training a visuomotor policy with cameras, and training legs - are named as such, with measured numbers and rented-GPU prices instead of hand-waving.

## Lessons

### What you are training, and what you train it on

1. [The data engine](https://robotics.biblio.guru/course/robot-learning/the-data-engine/) - a demonstration dataset is not a pile of recordings, it is the specification of everything the robot will ever be competent at, and you can audit one before you train on it.
2. [What a policy actually is](https://robotics.biblio.guru/course/robot-learning/what-a-policy-is/) - one function from observation to command, called thirty times a second with no memory, and you can build three of them without training anything.
3. [Behaviour cloning from scratch](https://robotics.biblio.guru/course/robot-learning/behaviour-cloning-from-scratch/) - delete the episode column, shuffle the rows, fit a network, and beat every untrained baseline in four seconds; the shuffle is the best idea in the lesson and the one that breaks next.

### The problem that makes this a subject

4. [Covariate shift: why the clone falls apart](https://robotics.biblio.guru/course/robot-learning/covariate-shift/) - a policy generates the states it is later judged on, so forty out of forty becomes zero out of forty when you change nothing but how long you let it run.
5. [Fixing covariate shift](https://robotics.biblio.guru/course/robot-learning/fixing-covariate-shift/) - every fix is the same move, expert labels on states the policy will actually visit, and they differ only in how well they guess where those are.
6. [Action chunking: deciding less often](https://robotics.biblio.guru/course/robot-learning/action-chunking/) - predicting a run of future actions cuts the number of decision points, which is what the error bound is quadratic in, and the bill arrives as seconds of blindness.

### The two architectures that shipped

7. [ACT, part by part](https://robotics.biblio.guru/course/robot-learning/act-in-depth/) - a transformer that turns one observation into the next hundred actions, read from the implementation rather than the abstract, including why the paper's own headline trick is off by default.
8. [Training ACT for real](https://robotics.biblio.guru/course/robot-learning/training-act/) - one real run on real SO-101 demonstrations, with measured wall-clock from a 2018 laptop and a result the author did not want.
9. [Two right answers, averaged into a wrong one](https://robotics.biblio.guru/course/robot-learning/multimodal-demonstrations/) - a squared-error loss asks for the average of everything the demonstrators did, and when they disagreed the average is a third thing that fails.
10. [Diffusion Policy: sample the action, do not summarise it](https://robotics.biblio.guru/course/robot-learning/diffusion-policy/) - learn to turn noise into an action, and every roll of the dice lands on a valid answer instead of the midpoint of all of them.
11. [Training a diffusion policy, and a fair fight against ACT](https://robotics.biblio.guru/course/robot-learning/training-diffusion-policy/) - three heads, one dataset, one budget, and the discovery that the policy with the worse validation loss is the one that reaches the goal.

### Saying it works, and meaning it

12. [What it takes to say a policy works](https://robotics.biblio.guru/course/robot-learning/evaluation-methodology/) - a success rate is an estimate with a denominator, an uncertainty and a protocol, and at ten trials the interval is wider than any difference worth arguing about.
13. [Building an eval harness you can trust](https://robotics.biblio.guru/course/robot-learning/building-an-eval-harness/) - a harness is a recording device for the protocol, not a loop that counts successes, and it is the file you carry into every module after this one.

### Learning from a score instead of a demonstration

14. [Scoring a robot instead of showing it](https://robotics.biblio.guru/course/robot-learning/reinforcement-learning-the-idea/) - reinforcement learning buys you tasks nobody can demonstrate and charges you a simulator, a hundred million failures, and a specification the optimiser reads adversarially.
15. [From a score to a gradient, and why PPO clips](https://robotics.biblio.guru/course/robot-learning/policy-gradients-to-ppo/) - no gradient passes through physics, so you nudge the probability of actions that scored well, and everything after that is fighting the noise in that estimate.
16. [Training a quadruped to walk](https://robotics.biblio.guru/course/robot-learning/training-locomotion/) - nobody writes the gait; it falls out of a weighted sum of penalties and enough parallel simulation, and this is the first thing in the course your laptop genuinely cannot do.
17. [Crossing to hardware: randomise, do not match](https://robotics.biblio.guru/course/robot-learning/sim-to-real/) - the fix for the reality gap is not a better simulator but a family of deliberately wrong ones wide enough that reality is one of them, and you can measure what that costs.

### The decision, and the proof

18. [Imitation, reinforcement, or just write the rule](https://robotics.biblio.guru/course/robot-learning/il-rl-or-scripted/) - three paradigms are three different artifacts a human has to author, the task decides which one you can produce, and the unit of decision is the segment rather than the project.

### Milestone

19. [ACT against Diffusion Policy, and the curve that matters more](https://robotics.biblio.guru/course/robot-learning/project-act-vs-diffusion/) - thirty training runs, fifteen hundred trials, and a writeup whose proudest sentence names what the trial count could not settle.

Each lesson has runnable companions in [`code/`](code/) (starters with `TODO(you)` markers) and full versions in [`solutions/`](solutions/). Figures live in [`assets/`](assets/): hand-authored SVG for the schematics, generated plots for everything that reports a measurement. The milestone scaffold is in [`project/`](project/) and deliberately has no solution file.

## Milestone project (summary; full spec in [lesson 19](https://robotics.biblio.guru/course/robot-learning/project-act-vs-diffusion/))

One artifact: a comparison that survives someone else re-running it.

- **The matrix.** Two tasks chosen to disagree - the groove world from Lesson 3.4, which has one right answer everywhere, and the corridor from Lesson 3.9, which has two - crossed with two policy families, two data budgets (10 and 50 demonstrations) and three seeds. Twenty-four cells, plus six ablation runs: chunk size for the ACT-style head, observation steps for the diffusion head.
- **The protocol.** Every cell scored on the identical seeded scenes in a rotating order, through the harness from Lesson 3.13, with the success predicate and failure tags frozen and fingerprinted before trial one. At least fifty trials per cell, appended to four log files that resume after a crash.
- **The writeup.** Seven sections, and the awkward ones are the point: per-arm rates with their Wilson intervals, seed spread quoted separately from trial spread, the ten-versus-fifty data-scaling curve, the ablations, the failure-tag histogram, and an explicit list of every pair your trial count could not separate.

`python act_vs_diffusion.py --check` reads the logs and the writeup and exits non-zero while any of that is missing. It cannot tell whether your success predicate is honest or whether you wrote the prose before you saw the numbers; those stay yours.

There is a laptop path and a rented-GPU path, and the starter never learns which you took. The laptop path trains the module's own chunked heads in seconds per cell and is genuinely enough to answer the question. The rented path runs real LeRobot on real datasets and is the one that transfers to Module 4, for single-digit dollars.

## Exit test

Explain cold, to Claude-as-examiner: why a held-out loss is a claim about frames and a success rate is a claim about episodes, and give the measurement that separates underfitting from distribution shift using one trained network and an expert; the single move that noise injection, recovery demonstrations and DAgger all make, and why one of them scored zero on the same label budget; why action chunking helps without claiming a theorem that does not exist, and what the chunk length costs in seconds at your control rate; what the latent inside ACT is for and why it is a vector of zeros on every robot step ACT has ever taken; why a squared-error loss cannot represent two valid answers, and why switching to absolute error does not rescue it; why diffusion's training objective is mean squared error without being the same mistake; PPO's clip described as what it is, a rate limiter that makes reusing a batch safe; and domain randomisation's mechanism stated without using the word "robust".

Then defend your milestone: which comparison your intervals actually settled, which shape your data-scaling curve took and what you would do about it, and which single number in your table you would drop if a reviewer had time to challenge only one.

## What expires here

More than anywhere else in the course so far. This module is dated throughout with `Field state` notes; the short version of what to re-check, and where:

- **LeRobot's shipped defaults**, which are the specification the lessons read as much as the papers are. `chunk_size`, `n_action_steps` and `temporal_ensemble_coeff` for ACT in Lesson 3.7; `horizon`, `n_action_steps` and `num_inference_steps` for Diffusion Policy in Lesson 3.10. Both tables were read at tag v0.6.1 on 2026-08-09 against the library's own config files, and the diffusion numbers already quadrupled once between releases. Re-check the config source for the version you install, not a tutorial.
- **The version boundaries that make numbers incomparable**: the v0.6.0 loss-normalisation fix for padded actions, and the v0.5.1-to-v0.6.x diffusion default change. Named in Lessons 3.8, 3.10 and 3.19. Check the release notes for anything newer.
- **GPU rental prices and job timeouts**, quoted in Lessons 3.8, 3.11, 3.16 and 3.19 with a read date of 2026-08-09. This market moves every few months; the arithmetic around the prices is what survives.
- **What a Mac cannot do**, in Lessons 3.8 and 3.16. Both claims are about missing backends rather than slow ones - PyTorch MPS on an Intel integrated GPU, and JAX on any Mac GPU - so they change only if a backend appears. Check the projects' own installation docs.
- **The frontier framing** in Lesson 3.7's closing section and Lesson 3.10's last note: chunking and generative action heads are now inside the large vision-language-action models, and flow matching is displacing the hundred-step sampler. Module 5 is where that gets picked up, and it is the part of this module most likely to read as dated first.

Each of those is stated in its lesson with the primary source beside it, so re-research is a bounded job rather than a re-read. Update [`resources/links.md`](../resources/links.md) while you are there.
