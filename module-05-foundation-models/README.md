# Module 5 - Foundation Models: VLAs and the frontier

**Goal:** stop admiring the frontier and start operating at it. By the end of this module you can draw the major vision-language-action architectures from a blank page, fine-tune an open one on the dataset you collected in Module 4, run a 3-billion-parameter open model on your own bench, and publish the comparison the ecosystem almost never does properly: one robot, one dataset, one protocol, classical imitation learning against a foundation model, with the numbers that went the wrong way left in.

**Two paths, inherited from Module 4.** This module runs with a physical SO-101 or without one, and the choice is the one you made in [Module 4 lesson 0](https://robotics.biblio.guru/course/hardware/the-simulation-path/). Without an arm, the frozen dataset is a pinned public LeRobot revision plus your simulated one, checkpoints are ranked in the simulator and offline on held-out real episodes rather than on a bench, and the milestone raises the trial count from twenty per cell to two hundred because trials cost compute instead of an evening. Lessons 10, 11, 12 and 15 carry a **Without hardware** section before their Check-yourself questions, and [lesson 18](https://robotics.biblio.guru/course/foundation-models/project-the-comparison/) states both acceptance criteria.

**Prerequisites:** Module 3 (you have trained ACT and built an eval harness) and Module 4 (you have a dataset and a policy that works on it, on a bench of either kind). The one thing this module genuinely assumes is a frozen dataset and an evaluation protocol you wrote yourself.

**Time:** ~8-10 weeks at course pace; about 9 hours of reading plus the runs. **Compute:** the heaviest of the course, and it is the real gate on both paths. A SmolVLA fine-tune fits a 24 GB consumer card; anything π₀-class wants rented A100-class hardware, and one lesson costs real money to complete. **Hardware:** optional throughout.

**Field state:** every model named here was checked on 9 August 2026. Dated claims carry a visible date on the page. Architectures and mechanisms age slowly; checkpoints, licences and prices age in months. Lesson 17 is the procedure for re-checking them yourself.

## Lessons

### What a foundation model is, and where it came from

1. [What pretraining actually buys a robot](https://robotics.biblio.guru/course/foundation-models/what-a-foundation-model-buys/) - a policy trained on your demonstrations knows your demonstrations, and there are three things a demonstration cannot contain no matter how many you record.
2. [The lineage: four ideas, not one scaling curve](https://robotics.biblio.guru/course/foundation-models/the-vla-lineage/) - between 2022 and 2024, four separable inventions turned a task-specific robot network into something you can download and talk to.
3. [How you write an action down decides everything else](https://robotics.biblio.guru/course/foundation-models/action-representations/) - pick a number from a list or produce it directly; that one choice sets your training cost, your control rate, and how much language ability survives.
4. [The fast/slow split: three clocks, not two models](https://robotics.biblio.guru/course/foundation-models/the-fast-slow-split/) - the deadlines belong to physics, not to taste, and no single network runs well at both ends of the range.

### Four architectures, read closely

5. [π₀ and π₀.₅: the open reference design](https://robotics.biblio.guru/course/foundation-models/pi-zero-in-depth/) - two experts in one attention stack, and the deliberate gradient cut that stops a fresh action head from sandblasting a pretrained backbone.
6. [GR00T and Gemini Robotics: a tensor or a sentence](https://robotics.biblio.guru/course/foundation-models/groot-and-gemini-robotics/) - two more ways to cut the same line, where what matters is not the architecture but what crosses the cut.
7. [Helix and the closed frontier](https://robotics.biblio.guru/course/foundation-models/helix-and-the-closed-frontier/) - the most impressive system in public view is the one you can check least; downloadable, published and demonstrated are three different grades of evidence.
8. [Comparative anatomy: three designs on one page](https://robotics.biblio.guru/course/foundation-models/comparative-anatomy/) - one skeleton, three variables, and the drawing you should be able to reproduce from memory. This is the module's exit test.

### Your own fine-tune

9. [SmolVLA: the one you can actually run](https://robotics.biblio.guru/course/foundation-models/smolvla-the-one-you-can-run/) - 450 million parameters, pretrained on community datasets from other people's spare rooms, and small enough that the whole loop closes on a card you already own.
10. [Preparing your dataset for a foundation model](https://robotics.biblio.guru/course/foundation-models/preparing-your-dataset/) - a VLA eats what ACT ate plus a sentence, and the experiment depends on freezing the dataset before anybody trains anything.
11. [Fine-tuning SmolVLA on your own robot](https://robotics.biblio.guru/course/foundation-models/fine-tuning-smolvla/) - the command is one line, the loss curve is beautiful, and neither tells you whether the policy works.
12. [Benchmarking your fine-tune against ACT](https://robotics.biblio.guru/course/foundation-models/benchmarking-against-act/) - twelve out of twenty against nine out of twenty is not a result, and knowing exactly why is the whole skill.
13. [When the big model loses](https://robotics.biblio.guru/course/foundation-models/when-the-big-model-loses/) - the published evidence says a pretrained VLA is not automatically better than a small policy trained from scratch, and it says so out loud.

### Running a frontier model, and reading the ones you cannot run

14. [openpi hands-on: a 3B model on your bench](https://robotics.biblio.guru/course/foundation-models/openpi-hands-on/) - a plumbing problem with a hard memory floor, a licence you have to read yourself, and a network hop inside your control loop.
15. [Running at the edge: latency, chunks and where the model lives](https://robotics.biblio.guru/course/foundation-models/running-at-the-edge/) - the forward pass is slower than the control period, always, and every deployment in this field is a scheme for hiding that.
16. [World models: a learned simulator, and what it is actually for](https://robotics.biblio.guru/course/foundation-models/world-models/) - predicting what happens next rather than what to do next, and where that currently earns its keep.
17. [Reading the frontier: triaging a new release in an hour](https://robotics.biblio.guru/course/foundation-models/reading-the-frontier/) - every model named in this module will be superseded; the procedure for deciding whether a new one matters will not be.

### Milestone

18. [The comparison nobody publishes](https://robotics.biblio.guru/course/foundation-models/project-the-comparison/) - one dataset, three policies, one protocol, four hundred real trials, and a public writeup that reports the result you did not want.

Figures live in [`assets/`](assets/) as hand-authored SVG, so they diff and review like any other source. The milestone harness is in [`project/`](project/): [`compare.py`](project/compare.py) supplies randomised interleaved trial ordering, a per-trial recorder and a Wilson confidence interval, and leaves the task definitions and the policy adapter to you. There is deliberately no solutions directory for this module. A solution would be somebody else's numbers on somebody else's bench, which is the exact thing that does not transfer.

## Milestone project (summary; full spec in [lesson 18](https://robotics.biblio.guru/course/foundation-models/project-the-comparison/))

ACT against a fine-tuned SmolVLA against a π₀-class model, with one frozen dataset and one protocol applied identically to all three. A task ladder that runs from the trained condition out to conditions nobody demonstrated, predictions registered in writing before the first run, and cost reported as a column rather than a footnote.

**Hardware criterion:** the ladder on the arm from Module 4, twenty trials per cell, 280 to 420 scored trials, and the honest sentence that twenty trials did not separate two policies wherever that is true.

**Simulation criterion:** the same four-task ladder built in the Module 2 scene, including the language cliff with coloured distractors; the no-pretraining ablation mandatory rather than optional; two hundred trials per cell for about 4,200 total, which is unattended overnight compute; intervals about seven points wide, which removes the "did not separate" sentence and obliges you to resolve the ten-point gaps; a second evaluation surface reporting offline chunk-level action error on held-out episodes of a pinned public SO-101 dataset, plus the rank correlation between the two surfaces; every perturbation axis listed explicitly, because you can only perturb what you coded.

The output on either path is this module's public writeup: architecture diagrams drawn from memory, honest intervals, the venue named in the headline, and the cells where the big model lost.

## Exit test

Draw π₀.₅, GR00T N1.7 and Helix from a blank page: backbone, action head, where the seam sits, what crosses it, and at what rate. Say which of the three you could download today and what licence you believe covers the weights, with the date you checked. Defend, from your own numbers, when a 50-demonstration ACT beats a fine-tuned VLA and when it is the other way round. State what pretraining bought your fine-tune, quantitatively, with an interval attached and a trial count under it.
