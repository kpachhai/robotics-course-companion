# Module 1 - Foundations: the language of robots

**Goal:** by the end of this module you read and write the core language of robotics - coordinate frames, rotations, transforms, kinematics, and feedback control - because you implemented every one of them yourself in plain Python. This is the module most AI-background people skip. Then, months later, they can't debug why their learned policy sends the gripper somewhere strange, because the bug is a frame convention. You will not have that problem.

**Prerequisites:** Module 0; comfortable Python; high-school trig (we rebuild everything else as needed).
**Time:** ~9 hours of reading, 6-8 weeks at course pace with the exercises and the project. **Dependencies:** numpy, scipy, matplotlib only. No robot, no GPU, no simulator.

## Lessons

### Frames and rotations

1. [Where things are, and what that even means](https://robotics.biblio.guru/course/foundations/where-things-are/) - a position is a reading taken against something else, and the something else has to be said out loud.
2. [Frames in 2D: naming them, and moving between them](https://robotics.biblio.guru/course/foundations/frames-in-2d/) - two named frames, one point, and the single line of arithmetic that converts either description into the other.
3. [Rotations in 2D: a table of where the axes land](https://robotics.biblio.guru/course/foundations/rotations-in-2d/) - a rotation matrix is nothing but the two places the unit axes end up, written side by side as columns.
4. [Rotations in 3D: three axes, one rule](https://robotics.biblio.guru/course/foundations/rotations-in-3d/) - the same 2D matrix acting in each pair of axes, and what SO(3) means.
5. [Rotation order matters: two turns are a sequence, not a set](https://robotics.biblio.guru/course/foundations/why-rotation-order-matters/) - the gap between the two orders is itself a rotation you can measure.
6. [Euler angles and gimbal lock: the map tears, not the territory](https://robotics.biblio.guru/course/foundations/euler-angles-and-gimbal-lock/) - the most readable way to write an orientation and the least safe way to compute with one.
7. [Quaternions: the four numbers robotics runs on](https://robotics.biblio.guru/course/foundations/quaternions/) - orientation with no seams, plus the two failure modes that cost a day each.

### Transforms and kinematics

8. [Homogeneous transforms: one box for the turn and the shift](https://robotics.biblio.guru/course/foundations/homogeneous-transforms/) - packing rotation and translation into a single 4x4 so apply, chain and invert all become matrix multiplication.
9. [Composing transform chains: reading a product without sign errors](https://robotics.biblio.guru/course/foundations/composing-transform-chains/) - world to base to joint to hand, with the subscripts as a type check on the order.
10. [Forward kinematics: joint angles to hand pose](https://robotics.biblio.guru/course/foundations/forward-kinematics/) - a pure function from angles to pose, and why that determinism is what lets a robot repeat anything.
11. [The 2-link arm](https://robotics.biblio.guru/course/foundations/the-2-link-arm/) - the smallest machine that still has every hard problem in robot kinematics inside it. This course keeps coming back to it.
12. [Workspace and reachability: what the arm can actually touch](https://robotics.biblio.guru/course/foundations/workspace-and-reachability/) - a ring with a hole in it, and why reaching a point differs from reaching it facing the right way.

### Velocity, singularities, and solving backwards

13. [The Jacobian: if I nudge this joint, where does the hand go?](https://robotics.biblio.guru/course/foundations/the-jacobian/) - a sensitivity table with one column per joint.
14. [Singularities: where the arm loses a direction](https://robotics.biblio.guru/course/foundations/singularities/) - poses where the Jacobian's columns line up and a solver starts commanding speeds no motor can deliver.
15. [Inverse kinematics: solving for the angles](https://robotics.biblio.guru/course/foundations/inverse-kinematics/) - the closed form, why it always has two answers, and the damped iteration for when no closed form exists.

### Making a motor arrive

16. [Feedback control: measuring the mistake instead of predicting it](https://robotics.biblio.guru/course/foundations/feedback-control/) - why a perfectly computed command still misses.
17. [PID: three questions to ask about one error](https://robotics.biblio.guru/course/foundations/pid/) - each term earned separately on a gravity-loaded joint.
18. [Trajectories: getting there smoothly rather than instantly](https://robotics.biblio.guru/course/foundations/trajectories/) - a controller never asks whether the setpoint was reasonable, so you have to feed it reachable ones.

### Milestone

19. [Scripted pick and place, and the width of its error budget](https://robotics.biblio.guru/course/foundations/project-pick-and-place/) - assemble the previous eighteen lessons into an arm that works, then measure exactly what breaks it.

Each lesson has runnable companions in [`code/`](code/) (starters with `TODO(you)` markers) and full versions in [`solutions/`](solutions/). Figures live in [`assets/`](assets/): schematics are hand-authored SVG, data plots regenerate from `tools/diagrams/generate.py`.

Lessons 5, 6, 9, 11, 13, 15 and 17 carry an interactive widget on the course site. The markdown holds a static figure in its place, so nothing is lost reading the files directly.

## Milestone project (summary; full spec in [lesson 19](https://robotics.biblio.guru/course/foundations/project-pick-and-place/))

A pure-Python 2D robot arm that picks a block from a random position and places it in a bin: your transforms library, your FK, your IK and a scripted state machine, rendered as a matplotlib animation. Then the part that makes it a real exercise: sweep perception noise until the grasp fails, and report the error band the machine can absorb. No learning anywhere; that is the point. This same two-link arm and its scripted tracker come back in Module 3 as the thing imitation and reinforcement are measured against, so the habit of reporting a band rather than a working demo starts here.

## Exit test

Explain cold, to Claude-as-examiner: why we use quaternions instead of Euler angles; what a Jacobian is, and its role in both IK and force control; what happens to a 2-link arm's Jacobian when the arm is fully stretched, and why that matters physically; why a P-only controller droops under gravity and which term fixes it. Then pass the grilling on your own project code.
