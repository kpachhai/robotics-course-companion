# Robotics, from scratch - companion

The runnable code and figures for [**Robotics, from scratch**](https://robotics.biblio.guru),
a free course that takes a working software engineer with no robotics background
from what a robot physically is through to the models on the frontier.

The course is 127 lessons and is free to read on the web. This repository holds
the parts you are meant to run and fork: the starter scripts, the worked
solutions, the module projects, and the figures.

## Layout

```
module-01-foundations/
  README.md      what the module covers, linking to the published lessons
  code/          starters - the file a lesson asks you to open and complete
  solutions/     the worked version, for after you have tried it
  project/       the module's project brief and scaffold
  assets/        the figures used in that module's lessons
resources/
  links.md       external references gathered across the course
```

Every module directory matches a module of the course, so a lesson that says
"open `module-01-foundations/code/fk_2link.py`" means exactly that path here.

## Running the code

Module 1 needs numpy, scipy and matplotlib and nothing else. Later modules add
their own dependencies, named in each module's README; none of Modules 0 to 3
needs a robot, a GPU or a simulator.

```sh
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy matplotlib
python module-01-foundations/code/fk_2link.py
```

## Licence

Two, because the code and the prose are not the same thing.

- **Code** - the `code/`, `solutions/` and `project/` directories - is
  [Apache 2.0](LICENSE). Fork it, ship it, no conditions beyond attribution and
  the patent grant.
- **Lessons, module READMEs and figures** are
  [CC BY-SA 4.0](LICENSE-CONTENT). Commercial reuse is permitted; derivatives
  must carry the same licence.

## Where the course itself lives

The lessons are published at [robotics.biblio.guru](https://robotics.biblio.guru)
and are free to read with no account and no tracking. The repository that builds
that site is private; this one is the public half, and it is generated from the
course repository rather than edited by hand, so please open issues here but
send prose corrections through the course site.

More courses at [biblio.guru](https://biblio.guru).
