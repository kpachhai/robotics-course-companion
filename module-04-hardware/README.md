# Module 4 - Real Hardware: the SO-101 and the data-quality loop

**Two paths, and you pick one before lesson 1.** This module can be done with a physical SO-101 or without one, and [lesson 0](https://robotics.biblio.guru/course/hardware/the-simulation-path/) is the honest comparison of the two. The short version: an arm supplies real data and a closed loop in one purchase, and without one you can still buy each separately - real SO-101 recordings are published on the Hugging Face Hub in the same format this module produces, and MuJoCo has been giving you a closed loop since Module 2. What no simulator supplies is calibration drift, lighting change, servo heating and the quality of a human demonstration, which is most of what makes real robotics hard. Every lesson below carries a short **Without hardware** section before its Check-yourself questions, and the milestone states both acceptance criteria in full.

**Goal:** stop being a simulation person, or find out precisely what that would mean. By the end of this module you have built an SO-101 leader-follower pair from a parts list (or read the same parts list against a model file), given it a calibration you can defend, driven it with your own hand, recorded a dataset, trained a policy on it, deployed that policy without breaking anything, diagnosed why it failed, and published a success rate with an interval attached that somebody else could check. Everything before this module could be undone with a keystroke. On the hardware path, nothing here can.

**Prerequisites:** Module 3. You need a trained ACT policy and an evaluation harness you wrote yourself, because this module reuses both and only changes where the data comes from. Module 2 matters more on the simulation path: its scene, its teleoperation rig and its recorder are the substitute bench. Module 1 pays off either way: the systematic reach error you predicted from a frame convention is the error you are about to measure.

**Time:** about 10 hours of reading, 6-8 weeks at course pace once the hardware arrives, or 4-6 weeks on the simulation path where nothing waits on shipping. **Hardware:** optional. The hardware path wants an SO-101 leader and follower pair (roughly $230 of parts self-sourced, more for a kit), two USB webcams, a bench you can clamp to, and controlled light; the simulation path wants none of it. **Software:** LeRobot, plus `numpy` for the analysis scripts and `opencv-python` for the two camera scripts; the simulation path adds nothing beyond `mujoco` from Module 2 and `pandas`, `pyarrow` and `huggingface_hub` for reading public datasets. **Compute:** the same GPU budget as Module 3; a sixty-episode dataset trains overnight on a consumer card.

**Field state:** every version, price and command in this module was checked on 9 August 2026 against LeRobot v0.6.1, and on 10 August 2026 the whole command surface - ten entry points, every flag, every servo register name and value - was re-checked against that release's *source*: `pyproject.toml`'s console scripts, the config dataclasses behind each flag, and the Feetech control table. Dated claims carry a visible date on the page. What that check cannot cover is behaviour: nothing here has been run on an arm, and where a claim depends on hardware the page says so. The mechanisms age slowly, the command names age in weeks, and lesson 3 tells you how to pin a version so your own work survives the next rename.

## Lessons

### Choosing a path

0. [Two ways to do this module](https://robotics.biblio.guru/course/hardware/the-simulation-path/) - an arm hands you real data and a closed loop at the same time; without one you can buy each separately, and the join is what you lose.

### The machine

1. [What is in the box, and what is not](https://robotics.biblio.guru/course/hardware/what-is-in-the-box/) - the SO-101 is a published bill of materials rather than a product, and every line in that list is a decision you are now responsible for.
2. [Assembling the arm](https://robotics.biblio.guru/course/hardware/assembling-the-arm/) - the build is mechanically easy and carries an ordering constraint that nothing on the parts enforces, so every expensive mistake is a gate you walked straight past.
3. [Servo IDs and the bus](https://robotics.biblio.guru/course/hardware/servo-ids-and-wiring/) - six motors share one wire, the id is the primary key on that wire, and every motor ships with the same one.
4. [Calibration: making two arms mean the same thing](https://robotics.biblio.guru/course/hardware/calibration/) - the map from raw encoder counts to joint angles, whose real job is agreement between everything that quotes those numbers rather than accuracy against the physical world.
5. [First motion: the smallest command that proves everything](https://robotics.biblio.guru/course/hardware/first-motion/) - one joint, five degrees, read back: every link from your Python process to a gear tooth exercised at once, with a separate signature for each way it can fail.
6. [Torque limits and safety: stop is not the same as hold](https://robotics.biblio.guru/course/hardware/torque-limits-and-safety/) - force on this arm is capped in registers rather than commanded, and the caps that ship set protect only the gripper - the one that would protect everything else ships switched off.

### The demonstration rig

7. [Teleoperation: your hand inside the loop](https://robotics.biblio.guru/course/hardware/teleoperation/) - six joint angles copied in one direction, tens of times a second, and nothing the follower touches ever travels back to your hand.
8. [What makes a good demonstration](https://robotics.biblio.guru/course/hardware/what-makes-a-good-demonstration/) - every second you teleoperate writes thirty rows of training data, so the quality of a demonstration is not whether it worked but whether it matches your other demonstrations.
9. [Cameras and viewpoints](https://robotics.biblio.guru/course/hardware/cameras-and-viewpoints/) - the camera rig is not equipment next to the robot, it is the observation space, and anything the cameras fail to resolve is unlearnable however good the model is.
10. [Hand-eye calibration, and why you are not going to do it](https://robotics.biblio.guru/course/hardware/hand-eye-calibration/) - a pixel is a ray, so turning one into a place needs depth and the camera-to-robot transform; an imitation policy skips both by never leaving pixel space, and charges you a repeatability discipline instead.
11. [Designing your first task](https://robotics.biblio.guru/course/hardware/designing-your-first-task/) - a task is not a sentence, it is the set of scenes that sentence has to work in, and your first one should be small enough that sixty demonstrations cover it.

### The loop

12. [Recording a real dataset](https://robotics.biblio.guru/course/hardware/recording-a-real-dataset/) - sixty demonstrations is an afternoon of deliberate repetition, and the most important key on the keyboard is the one that throws an episode away.
13. [Training on real data](https://robotics.biblio.guru/course/hardware/training-on-real-data/) - mostly arithmetic on frames, batches and epochs, plus one widely-copied scheduler flag that aborts an ACT run outright and one number that does not mean what you want it to.
14. [Deploying to the arm](https://robotics.biblio.guru/course/hardware/deploying-to-the-arm/) - a supervised experiment with a short duration, a motion clamp and a hand near the power plug, where cutting the power does not stop the arm, it drops it.
15. [The data-quality loop: where the next twenty episodes go](https://robotics.biblio.guru/course/hardware/the-data-quality-loop/) - a trained policy is a compressed copy of your demonstrations, so improving it is a measurement problem: watch it fail, sort the failures, aim the next episodes at the biggest bucket.

### Diagnosis and publication

16. [Debugging by symptom: what the arm is telling you](https://robotics.biblio.guru/course/hardware/debugging-by-symptom/) - a failure that repeats identically is a geometry problem, a failure that lands somewhere new each time is a data problem, and the two live in different parts of the building.
17. [Publishing hardware work: the number, the interval and the failure reel](https://robotics.biblio.guru/course/hardware/publishing-hardware-work/) - a robot result nobody can audit is a screenshot, and the five artifacts that make yours auditable cost an hour between them.

### Milestone

18. [A real robot doing a real task, twenty times, on camera](https://robotics.biblio.guru/course/hardware/project-a-real-robot-doing-a-real-task/) - one task taken from nothing to a measured, published success rate, with an evaluation a stranger could check.

Figures live in [`assets/`](assets/) as hand-authored SVG, so they diff and review like any other source. The analysis scripts are in [`code/`](code/): all of them run without the robot plugged in, most on the standard library alone, and each one exists because a lesson makes a claim that you should be able to check rather than believe. The milestone harness is in [`project/`](project/): [`run_eval.py`](project/run_eval.py) deals initial conditions in a fixed random order and appends one trial per line as you go, [`eval_report.py`](project/eval_report.py) turns that log into a rate, an interval and a failure histogram, and neither of them talks to the robot on purpose. There is no solutions directory for this module. A solution would be somebody else's numbers on somebody else's bench, and that is exactly the thing that does not transfer.

## Milestone project (summary; full spec in [lesson 18](https://robotics.biblio.guru/course/hardware/project-a-real-robot-doing-a-real-task/) and [`project/README.md`](project/README.md))

**Hardware criterion.** Pick a task you can already teleoperate ten times out of ten. Freeze the rig, record around fifty episodes against a coverage plan, train, deploy, and take the result round the data-quality loop until it works at least seven times in ten. Then prove it: twenty consecutive uncut trials in one sitting, with the protocol written down before the first one, the camera running throughout, and every void declared. The output is a public writeup: dataset and checkpoint on the Hub, a model card naming the LeRobot version and the camera positions, a photograph of the rig, the uncut video, a failure reel with every failure tagged, and a lead sentence carrying the interval rather than the point estimate.

**Simulation criterion.** The same task, in the Module 2 scene, with a scene file frozen and hashed. Two datasets of fifty episodes each on the same schedule, one teleoperated by your hand and one from the scripted controller, so that demonstration consistency becomes a measurement rather than a thing you lost. Two hundred scored trials per policy from a seed never trained on, a success predicate committed before training, and a bar of 70% with the lower bound of the Wilson interval above 60%. Then the robustness table: the same protocol re-run under four perturbations you never trained on, one at a time. Then touch real data once, by training the same architecture on a pinned public SO-101 dataset and reporting chunk-level action error on a held-out split, with a paragraph on why that is not a success rate. Publish the same artifacts plus the scene hash, the simulator version and every seed, and put the venue in the headline sentence.

Neither column is the easy one. The hardware reader is limited by wall-clock and pays for it in a wide interval; the simulation reader is limited by nothing and therefore owes a narrow one.

## Exit test

Say what the two calibration numbers per joint actually change, and predict the precise shape of the error a wrong one produces. Explain why the leader mixes three gear ratios and the follower does not. State what crosses the gap between leader and follower during teleoperation and what never does. Given a policy that reaches three centimetres left on every single trial, name the layer to check first and the one probe that isolates it; then do the same for a policy that fails somewhere new each time. Quote your own success rate with its 95% interval, its trial count and the bench that produced it, and say out loud what that interval does not entitle you to claim. On the simulation path, add one: name the four properties of a real arm your number is silent about, and say which of this module's lessons exists because of each.
