# Module 2 - Simulation: MuJoCo, your infinite lab

**Goal:** by the end of this module you can build a MuJoCo scene, drive a simulated SO-101 with the kinematics you wrote yourself, plan a collision-free route through a space you built yourself, script a pick-and-place that works, and record the result as a dataset something else can train on. The reason to do this before touching hardware is arithmetic: on a real arm an attempt costs twelve seconds and a human to reset it, and in simulation it costs milliseconds and a function call. The reason to be suspicious of it is the same arithmetic. Every number in the model is a guess, and this module is as much about knowing which guesses you are trusting as it is about the tool.

**Prerequisites:** Module 1. You need your own transforms, forward kinematics, Jacobian and inverse kinematics, because Lesson 2.9 points them at a real robot and diffs them against the engine.
**Time:** ~10 hours of reading across the seventeen lessons, plus about four hours on the milestone; 3-4 weeks at course pace with the exercises. **New dependency:** `mujoco`, installed in Lesson 2.3, which is the only setup step in the module. **Hardware:** none. Everything runs on a laptop CPU, and no exercise before the milestone takes more than a minute.

## Lessons

### What a simulator is, and getting one running

1. [Why simulate](https://robotics.biblio.guru/course/simulation/why-simulate/) - simulation is a staging environment for physics: attempts cost nothing, reset is a function call, and every number in it is a guess you eventually pay for.
2. [What a physics engine actually does](https://robotics.biblio.guru/course/simulation/what-a-simulator-does/) - every step, the engine guesses where everything will be, finds every place that guess is illegal, and solves one system for the forces that make it legal again.
3. [Installing MuJoCo, and the one thing macOS will not let you do](https://robotics.biblio.guru/course/simulation/installing-mujoco/) - one pip command, a scripted simulation that checks its own physics, and the `mjpython` rule that stops every macOS user exactly once.

### Describing a robot

4. [MJCF anatomy: a robot is a tree with indentation](https://robotics.biblio.guru/course/simulation/mjcf-anatomy/) - the scene format nests bodies inside bodies, and that nesting is the transform chain you composed by hand in Module 1.
5. [Reading somebody else's robot](https://robotics.biblio.guru/course/simulation/reading-the-arm-model/) - four minutes with the SO-101 model file gets you its degrees of freedom, its joint limits, its servo gains and one thing it can never do.

### Driving it from Python

6. [The simulator is a state machine you advance](https://robotics.biblio.guru/course/simulation/driving-the-sim-from-python/) - `mjModel` holds what never changes, `mjData` holds what does, and `mj_step` turns one into the next.
7. [Actuators: what you may say, and what you may know](https://robotics.biblio.guru/course/simulation/actuators-and-control-modes/) - the actuator type decides what the number in `data.ctrl` means, and position control turns force from something you command into something you can only watch.
8. [Sensors and cameras: you decide what the robot is allowed to see](https://robotics.biblio.guru/course/simulation/sensors-and-cameras/) - the model ships with no sensors and perfect knowledge, so every number a policy may read is a line you wrote, and the easiest numbers to reach for are the ones no hardware can produce.

### Your code against the engine

9. [Your kinematics meets theirs](https://robotics.biblio.guru/course/simulation/your-kinematics-meets-theirs/) - point your Module 1 maths at the SO-101, subtract MuJoCo's answer, and find the geometry agreeing to the last bit while the physics does not.
10. [Teleoperating the sim](https://robotics.biblio.guru/course/simulation/teleoperating-the-sim/) - six keys steer a point in space, one IK step per physics tick drags the arm after it, and the demonstrations are exactly as bad as the interface that produced them.

### Turning an arm into a task

11. [Building a task scene](https://robotics.biblio.guru/course/simulation/building-a-task-scene/) - geometry decides whether the task is possible at all, and a success predicate decides whether an episode counted, with no human in the loop.
12. [Contact and grasping: the parameters you did not write](https://robotics.biblio.guru/course/simulation/contact-and-grasping/) - a grasp is a sustained force balance across a dozen contact points, governed by settings that mostly came from somebody else's XML.

### Getting there without hitting anything

13. [Configuration space: the arm is a point](https://robotics.biblio.guru/course/simulation/configuration-space/) - every pose is one point in a space with one axis per joint, and an obstacle in the world becomes a forbidden region in it whose shape is nothing like the obstacle's.
14. [Sampling-based planning](https://robotics.biblio.guru/course/simulation/sampling-based-planning/) - free space stops being enumerable at about four joints, so planners stop examining it and start sampling it; RRT from scratch, PRM briefly, and exactly which guarantees that trade costs you.
15. [When planners beat policies, and when they do not](https://robotics.biblio.guru/course/simulation/when-planners-beat-policies/) - a planner needs a model and hands you a route you can check before anything moves, a policy needs demonstrations and works where the model runs out, and every real system uses both.

### Making the data

16. [Domain randomisation: vary what you got wrong, measure what you changed](https://robotics.biblio.guru/course/simulation/domain-randomization/) - randomisation is judged by the spread of the data it produces rather than by the success rate, and the evidence on what transfers is narrower than the folklore.
17. [Recording episodes something else can learn from](https://robotics.biblio.guru/course/simulation/recording-datasets/) - a dataset is a contract with a model you have not written yet, and almost everything that breaks it is decided in the fifteen lines that write one frame.

### Milestone

18. [A hundred episodes, and fifty that replay](https://robotics.biblio.guru/course/simulation/project-scripted-pick-and-place/) - two artifacts and one honest number, measured over a region you name out loud.

Each lesson has runnable companions in [`code/`](code/) (starters with `TODO(you)` markers) and full versions in [`solutions/`](solutions/). Figures live in [`assets/`](assets/) as hand-authored SVG. The milestone scaffold is in [`project/`](project/) and deliberately has no solution file.

## Milestone project (summary; full spec in [lesson 18](https://robotics.biblio.guru/course/simulation/project-scripted-pick-and-place/))

Two deliverables and a page about them.

- **An evaluation.** Run your scripted pick-and-place 100 times with the cube sampled from a box you state in metres, at a stated seed, judged by the success predicate from Lesson 2.11. Report the rate at three or four box widths rather than one, because a single rate hides which half of the region is carrying it. The bar is 90% at the nominal width; missing it with a labelled failure taxonomy is a better outcome than clearing it with no idea why.
- **A dataset.** Fifty successful episodes in the LeRobot v3.0 frame schema, which passes its own schema and timestamp check and, when replayed back through the simulator, lands the cube on the same numbers the original run did.

Then the writeup: the table, the box, the seed, the failure labels with counts, and an honest paragraph on what a learned policy would have to do that your scripted expert does not. Keep both artifacts. Module 3 works directly on this dataset - auditing it, cloning it, and running genuine DAgger against the scripted expert that produced it - and the milestone's rented-GPU path takes it as one of its two tasks. All of that only means anything because this run was measured properly.

## Exit test

Explain cold, to Claude-as-examiner: what a contact solver computes and why contacts cannot be resolved one at a time; how MJCF nesting maps onto a transform chain, and what the compiler filled in that you never typed; why a position actuator makes the force it applies readable in simulation and unreadable on hardware; what a configuration space is, why a round obstacle does not make a round hole in it, and what a sampling planner gives up in exchange for surviving six joints; what a success predicate has to contain beyond "the object is near the target", and which fake success each clause rejects; and why a success rate without its sampling region and trial count is not a number. Then defend your milestone: why your rate falls off at the widest box, and which of your failure labels is geometry rather than luck.

## What expires here

Less than in the modules ahead, but not nothing. The pinned MuJoCo version in Lesson 2.3 and the `mjpython` requirement are current as of the dated `Field state` notes in the lessons; the LeRobot dataset schema in Lesson 2.17 is v3.0 and is the one thing here most likely to move. Re-check both against their primary sources before Module 3, and update [`resources/links.md`](../resources/links.md) while you are there.
