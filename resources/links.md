# Canonical links

Curated hard: if it is here, it earned a place. Check dates before trusting anything fast-moving, VLA-related links especially. Last verified: **2026-08**.

## Courses and texts

- [MIT 6.4210 Robotic Manipulation](https://manipulation.csail.mit.edu) (Tedrake) - free notes, lectures and exercises; Module 1's deeper companion.
- [Underactuated Robotics](https://underactuated.mit.edu) (Tedrake) - the locomotion track's bible, and where the LQR and MPC that Lesson 1.17 names but does not teach actually live.
- Modern Robotics (Lynch and Park) - free PDF plus video lectures; the rigorous classical reference.
- [Hugging Face Robotics Course](https://huggingface.co/learn/robotics-course) - free and LeRobot-based; units were still rolling out as of August 2026.
- [Robot Learning: A Tutorial](https://huggingface.co/spaces/lerobot/robot-learning-tutorial) - the short version of Modules 3 and 5, from the LeRobot team.
- [Berkeley CS285 Deep RL](https://rail.eecs.berkeley.edu/deeprlcourse) - the reinforcement learning depth this course stops short of.

## Code and stacks

- [LeRobot](https://github.com/huggingface/lerobot) - the stack this course runs on, with its own [documentation](https://huggingface.co/docs/lerobot).
- [MuJoCo](https://mujoco.readthedocs.io) - the simulator from Module 2, plus [Menagerie](https://github.com/google-deepmind/mujoco_menagerie) for robot models and [Playground](https://github.com/google-deepmind/mujoco_playground) for tasks. The [release notes](https://github.com/google-deepmind/mujoco/releases) are the trigger for Module 2's re-research pass; minor releases land roughly monthly and do carry breaking changes.
- [openpi](https://github.com/Physical-Intelligence/openpi) - Physical Intelligence's release of the π0 and π0.5 weights and code.
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab) - GPU-parallel simulation; the locomotion track's tooling.
- [rerun](https://rerun.io) and [Foxglove](https://foxglove.dev) - the two logging and visualisation tools compared in Lesson 7.7.

## Papers map

- [keon/awesome-physical-ai](https://github.com/keon/awesome-physical-ai) - the maintained map of VLA, world-model and embodied-AI papers.
- Reading order for Module 5: RT-1 → RT-2 → Open X-Embodiment → OpenVLA → π0 → π0.5 → GR00T N1 ([arXiv:2503.14734](https://arxiv.org/abs/2503.14734)) → SmolVLA → the Gemini Robotics and Helix technical reports.

## Hardware

- SO-101 build guides - via the LeRobot documentation, which carries assembly, calibration and teleoperation.
- Kits: Seeed Studio (SO-ARM101), WowRobo, Hiwonder, Robonine, PartaBot and others. See the vendor list in the SO-ARM100 repository and price your own region. Lessons 0.8 and 4.1 price the official bill of materials at **$229.88 for the leader-follower pair** ($121.94 for a follower alone), excluding printing, cameras and a power supply, and put a realistic two-arm setup with webcams at **$300 to $400** if you do not own a 3D printer. The widely repeated "$100 arm" is 2025 launch coverage, not a 2026 purchasable.
- [Jetson AI Lab](https://www.jetson-ai-lab.com) - edge deployment recipes, including the openpi-on-Thor tutorial.

## Community and staying current

- LeRobot Discord - reachable from the LeRobot GitHub or documentation. The scene's living room, and where hackathons are announced.
- r/robotics, The Robot Report, and IEEE Spectrum's robotics section.
- Conferences whose talks go online: CoRL, RSS, ICRA, IROS, Humanoids.
- Follow-list seed: the LeRobot and Hugging Face robotics team, Physical Intelligence, Ted Xiao, Chris Paxton, Sergey Levine, Jim Fan (NVIDIA GEAR), Karol Hausman.
