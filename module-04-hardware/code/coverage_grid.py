"""Plan and audit the coverage of a real demonstration dataset (Lesson 4.15).

Two jobs, both pure arithmetic on your own recording plan:

  plan   emit a shuffled placement schedule so every cell of the workspace
         gets the same number of episodes, and you never drift toward the
         middle of the table without noticing.

  audit  read back where the object actually was in each recorded episode
         and report per-cell counts, empty cells, and the imbalance ratio.

No hardware and no dependencies: standard library only, so it runs on the
laptop, on the training box, and in CI. It knows nothing about LeRobot; you
feed it the placements, it feeds you the schedule.

Run:
    python coverage_grid.py plan  --episodes 60 --cols 3 --rows 2
    python coverage_grid.py plan  --episodes 60 --seed 7 --csv plan.csv
    python coverage_grid.py audit --csv recorded.csv --cols 3 --rows 2

The audit CSV wants a header and one row per episode:
    episode,x_cm,y_cm
    0,7.5,4.0
"""

import argparse
import csv
import random
import sys

# Workspace rectangle in centimetres, measured on YOUR table, front-left corner
# at the origin. Change these two lines before your first recording session.
WORKSPACE_W_CM = 30.0
WORKSPACE_H_CM = 20.0


def cell_bounds(col, row, cols, rows, width, height):
    """Return (x0, x1, y0, y1) of one grid cell in centimetres."""
    cw, ch = width / cols, height / rows
    return col * cw, (col + 1) * cw, row * ch, (row + 1) * ch


def cell_of(x_cm, y_cm, cols, rows, width, height):
    """Which cell a placement falls in. Returns None if outside the rectangle."""
    if not (0 <= x_cm <= width and 0 <= y_cm <= height):
        return None
    col = min(int(x_cm / (width / cols)), cols - 1)
    row = min(int(y_cm / (height / rows)), rows - 1)
    return col, row


def plan(episodes, cols, rows, width, height, seed, margin_cm):
    """Balanced, shuffled schedule of (episode, col, row, x_cm, y_cm)."""
    rng = random.Random(seed)
    cells = [(c, r) for r in range(rows) for c in range(cols)]
    per_cell, remainder = divmod(episodes, len(cells))
    if remainder:
        print(
            f"warning: {episodes} episodes does not divide evenly into "
            f"{len(cells)} cells; {remainder} cell(s) get one extra",
            file=sys.stderr,
        )
    order = cells * per_cell + rng.sample(cells, remainder)
    rng.shuffle(order)

    schedule = []
    for episode, (col, row) in enumerate(order):
        x0, x1, y0, y1 = cell_bounds(col, row, cols, rows, width, height)
        x = rng.uniform(x0 + margin_cm, x1 - margin_cm)
        y = rng.uniform(y0 + margin_cm, y1 - margin_cm)
        schedule.append((episode, col, row, round(x, 1), round(y, 1)))
    return schedule


def audit(rows_in, cols, rows, width, height):
    """Count placements per cell. Returns (counts dict, outside list)."""
    counts = {(c, r): 0 for r in range(rows) for c in range(cols)}
    outside = []
    for record in rows_in:
        cell = cell_of(record["x_cm"], record["y_cm"], cols, rows, width, height)
        if cell is None:
            outside.append(record)
        else:
            counts[cell] += 1
    return counts, outside


def print_grid(counts, cols, rows):
    """Top row of the printed grid is the far edge of the table, as you see it."""
    for row in reversed(range(rows)):
        line = "  ".join(f"{counts[(c, row)]:4d}" for c in range(cols))
        print(f"  row {row}  {line}")
    print("           " + "  ".join(f"col{c:<1d}" for c in range(cols)))


def read_placements(path):
    with open(path, newline="") as handle:
        for record in csv.DictReader(handle):
            yield {
                "episode": int(record["episode"]),
                "x_cm": float(record["x_cm"]),
                "y_cm": float(record["y_cm"]),
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["plan", "audit"])
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--width", type=float, default=WORKSPACE_W_CM)
    parser.add_argument("--height", type=float, default=WORKSPACE_H_CM)
    parser.add_argument("--margin", type=float, default=1.0,
                        help="keep placements this far inside a cell edge (cm)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", help="output file for plan, input file for audit")
    args = parser.parse_args()

    if args.mode == "plan":
        schedule = plan(args.episodes, args.cols, args.rows,
                        args.width, args.height, args.seed, args.margin)
        header = ("episode", "col", "row", "x_cm", "y_cm")
        if args.csv:
            with open(args.csv, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(header)
                writer.writerows(schedule)
            print(f"wrote {len(schedule)} placements to {args.csv}")
        else:
            print(",".join(header))
            for line in schedule:
                print(",".join(str(field) for field in line))
        return

    if not args.csv:
        parser.error("audit needs --csv pointing at your recorded placements")
    records = list(read_placements(args.csv))
    counts, outside = audit(records, args.cols, args.rows, args.width, args.height)

    print(f"{len(records)} episodes over {args.cols}x{args.rows} cells")
    print_grid(counts, args.cols, args.rows)

    values = list(counts.values())
    empty = [cell for cell, n in counts.items() if n == 0]
    print(f"\nmin {min(values)}  max {max(values)}  "
          f"imbalance {max(values) / max(min(values), 1):.1f}x")
    if outside:
        print(f"{len(outside)} placement(s) outside the declared workspace")
    if empty:
        print(f"EMPTY CELLS: {sorted(empty)} -- the policy has never seen these")
    elif max(values) > 2 * max(min(values), 1):
        print("one cell has more than twice another; record into the thin cells next")
    else:
        print("coverage is even enough; spend the next episodes on a new axis")


if __name__ == "__main__":
    main()
