# Module 7 - Specialization, Capstone, and Positioning

**Goal:** convert a year of general skill into a position. One track chosen on evidence rather than appetite, the shared systems literacy the industry assumes you already have, one capstone that settles one falsifiable claim, and the writing and packaging that let a stranger evaluate all of it without you in the room.

**Prerequisites:** Modules 0 through 6. This module is deliberately last, because its first lesson spends nine months of evidence about what actually pulled you, and that evidence does not exist earlier.

**Time:** about 5.2 hours of reading across eleven lessons, then six to eight weeks of capstone work that the lessons scope but do not contain. **Compute:** none beyond what your chosen track needs.

**Field state:** the compensation, tooling and hiring claims in lessons 1, 6 and 11 were checked in August 2026 and carry their dates in the text. Lesson 3 deliberately quotes no hardware prices at all, for the same reason. Method ages slowly; job markets and distro support windows age in months. Re-check anything dated before you act on it.

## Lessons

### Choosing

1. [Choosing a track without lying to yourself](https://robotics.biblio.guru/course/specialization/choosing-a-track/) - three tracks, four criteria, and nine months of evidence you already collected about which one is actually yours.
2. [Track A: agentic robotics](https://robotics.biblio.guru/course/specialization/track-agentic-robotics/) - everything you know about tool-use architecture holds, right up to the point where a failed call leaves the world changed.
3. [Track B: locomotion and humanoids](https://robotics.biblio.guru/course/specialization/track-locomotion/) - the track where the physics is genuinely harder, the bill arrives in GPU hours and broken hardware, and almost none of your background compounds.
4. [Track C: data and fleet infrastructure](https://robotics.biblio.guru/course/specialization/track-data-and-fleet/) - the least glamorous track, the most employable one, and the only one where nearly all of your existing background transfers without translation.
5. [The machine-economy thread](https://robotics.biblio.guru/course/specialization/the-machine-economy-thread/) - robots holding accounts, paying each other for metered resources, and proving what they did to someone who was not there, treated as early and mostly unbuilt.

### Shared systems literacy, whichever track you took

6. [ROS 2: the honest minimum](https://robotics.biblio.guru/course/specialization/ros2-the-honest-minimum/) - you built a working robot without typing `ros2` once; here is exactly how much of it you owe the industry, and how much of what is written about it is already dead.
7. [Logging and visualisation: the run is gone, the recording is not](https://robotics.biblio.guru/course/specialization/logging-and-visualisation/) - a robot failure cannot be reproduced, so the recording is the only evidence that will ever exist. Learn the format deeply and the viewers loosely.

### The capstone, and what it becomes

8. [Designing a capstone that proves a position](https://robotics.biblio.guru/course/specialization/designing-your-capstone/) - a capstone is an argument with a robot attached, and the argument has to be settled before the first commit.
9. [Shipping it like a product](https://robotics.biblio.guru/course/specialization/shipping-it-like-a-product/) - the version of your capstone that works with you absent, which is the only version most people will ever see.
10. [Writing that travels](https://robotics.biblio.guru/course/specialization/writing-that-travels/) - the piece that gets read is the page somebody was already looking for, and in robotics the missing page is almost always the one with the failures in it.
11. [Positioning, and the finish line](https://robotics.biblio.guru/course/specialization/positioning-and-the-finish-line/) - a year of work becomes a position when a stranger can check it in seven minutes without you in the room.

Figures live in [`assets/`](assets/) as hand-authored SVG, so they diff and review like any other source. This module ships no starter code and no solutions directory, and that is the point: the artifact is your capstone, on your chosen track, and there is nothing here for a reference implementation to be a reference to.

## Capstone (summary; full spec in [lesson 8](https://robotics.biblio.guru/course/specialization/designing-your-capstone/))

One project, six to eight weeks, built around a single falsifiable claim written down before the first commit, with the evidence that would settle it named at the same time. Weeks ordered by risk rather than by comfort, so the part most likely to kill the project is attempted first. A spine that must work and a shell that may be cut. Numbers frozen before the results exist, in a protocol you would still run if it made you look worse. It ships as a README a stranger can read in ten minutes, an uncut video with metrics on screen, and a limitations section written before somebody else writes it for you.

## Exit test

State your track and defend it on pull, compounding, burn and market shape, including what you gave up. Explain the ROS 2 concepts a working engineer is assumed to know, and tell current practice from legacy at a glance. Say what belongs in a recording so that any failed trial is answerable a month later, and which viewer you would open for which question. Then hand someone your capstone README and video, say nothing, and see whether they can name your claim, your evidence and your two biggest limitations in seven minutes. That last one is the actual exit test for the whole course.
