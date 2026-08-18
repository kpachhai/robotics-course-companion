"""Lesson 2.14 - sampling-based planning for the 2-link arm (starter).

An RRT and a PRM over the configuration space built in the previous lesson.
Neither one ever builds the forbidden set. Both only ask one question, many
times: is this configuration free, and is this short move between two of them
free.

Run:  python rrt_2link.py             one RRT run, then 40 seeds, then shortcutting
      python rrt_2link.py --prm       the roadmap, built once and queried many times
      python rrt_2link.py --grid      why the exhaustive answer stops being available
      python rrt_2link.py --nopath    what the planner says when there is no path
      python rrt_2link.py --audit     the collision the edge checker stepped over
      python rrt_2link.py --touch     what happens when the goal is to touch something
"""
import sys
import time

import numpy as np

from cspace_2link import BENCH, EXTRA_DISC, GOAL, START, TAU, collides

# A configuration in the pocket that EXTRA_DISC seals off. Reachable in the sense
# of Module 1's workspace lesson, and not connected to START by any path.
POCKET = (-0.50, -1.83)

STEP = 0.25          # how far one extension moves, in radians
EDGE_RES = 0.05      # an edge is checked every 0.05 rad of joint travel
GOAL_BIAS = 0.05     # how often the random sample is replaced by the goal
GOAL_TOL = 0.25      # close enough to connect straight to the goal
MAX_NODES = 20_000


def wrap(delta):
    """Shortest signed angular difference, componentwise. The space is a torus."""
    return (delta + np.pi) % TAU - np.pi


def distance(a, b):
    return float(np.linalg.norm(wrap(np.asarray(b) - np.asarray(a))))


def edge_free(a, b, obstacles=BENCH, resolution=EDGE_RES):
    """Check the straight move from a to b at a fixed spacing.

    Returns (free, checks). This is where a planner spends nearly all of its time.
    """
    # TODO(you) 1: walk the wrapped difference from a to b in steps no larger
    # than `resolution` radians, calling collides() at each one, and count the
    # calls. Return on the first collision - a planner that checks the whole edge
    # after already failing is throwing away most of its budget.
    raise NotImplementedError


def path_length(path):
    return sum(distance(path[i], path[i + 1]) for i in range(len(path) - 1))


def rrt(start, goal, obstacles=BENCH, seed=0, max_nodes=MAX_NODES,
        step=STEP, goal_bias=GOAL_BIAS, resolution=EDGE_RES):
    """Grow one tree from the start until an extension lands within GOAL_TOL of the goal.

    Returns a dict: path (or None), nodes, samples, checks, seconds.
    """
    rng = np.random.default_rng(seed)
    store = np.empty((max_nodes, 2))       # grown in place; rebuilding it every
    store[0] = np.asarray(start, float)    # iteration is what makes a naive RRT quadratic
    nodes = store[:1]
    parent = [-1]
    samples = checks = 0
    t0 = time.perf_counter()

    while len(nodes) < max_nodes:
        samples += 1
        if rng.random() < goal_bias:
            target = np.asarray(goal, float)
        else:
            target = rng.uniform(-np.pi, np.pi, 2)

        # TODO(you) 2: the extension. Find the index `near` of the tree node
        # closest to `target` under the wrapped distance, then set `new` to the
        # configuration `step` radians from that node in the target's direction -
        # or `target` itself if it is already nearer than `step`. Two lines of
        # numpy. This is the whole of RRT; everything else is bookkeeping.
        near = 0
        new = nodes[0]
        raise NotImplementedError

        free, used = edge_free(nodes[near], new, obstacles, resolution)
        checks += used
        if not free:
            continue

        store[len(nodes)] = new
        nodes = store[: len(nodes) + 1]
        parent.append(near)

        if distance(new, goal) < GOAL_TOL:
            free, used = edge_free(new, goal, obstacles, resolution)
            checks += used
            if free and len(nodes) < max_nodes:
                store[len(nodes)] = np.asarray(goal, float)
                nodes = store[: len(nodes) + 1]
                parent.append(len(nodes) - 2)
                path = []
                i = len(nodes) - 1
                while i != -1:
                    path.append(nodes[i].copy())
                    i = parent[i]
                path.reverse()
                return {"path": path, "nodes": len(nodes), "samples": samples,
                        "checks": checks, "seconds": time.perf_counter() - t0}

    return {"path": None, "nodes": len(nodes), "samples": samples,
            "checks": checks, "seconds": time.perf_counter() - t0}


def shortcut(path, obstacles=BENCH, iterations=200, seed=0, resolution=EDGE_RES):
    """Repeatedly try to replace a stretch of the path with the straight line across it."""
    # TODO(you) 3: pick two waypoint indices at random, at least two apart. If
    # the straight move between them is free, delete everything in between.
    # Repeat `iterations` times. Keep the endpoints.
    raise NotImplementedError


def prm(obstacles=BENCH, n_samples=600, k=12, seed=0, resolution=EDGE_RES):
    """Build a roadmap once: sample free configurations, connect each to its k nearest."""
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    nodes = []
    while len(nodes) < n_samples:
        q = rng.uniform(-np.pi, np.pi, 2)
        if not collides(q, obstacles):
            nodes.append(q)
    nodes = np.array(nodes)

    edges = {i: [] for i in range(len(nodes))}
    kept = tested = 0
    for i in range(len(nodes)):
        order = np.argsort(np.linalg.norm(wrap(nodes - nodes[i]), axis=1))[1: k + 1]
        for j in order:
            j = int(j)
            if j < i:
                continue
            tested += 1
            free, _ = edge_free(nodes[i], nodes[j], obstacles, resolution)
            if free:
                kept += 1
                d = distance(nodes[i], nodes[j])
                edges[i].append((j, d))
                edges[j].append((i, d))
    return {"nodes": nodes, "edges": edges, "kept": kept, "tested": tested,
            "seconds": time.perf_counter() - t0}


def prm_query(roadmap, start, goal, obstacles=BENCH, k=12, resolution=EDGE_RES):
    """Attach start and goal to the roadmap, then run Dijkstra over it."""
    import heapq

    t0 = time.perf_counter()
    nodes = roadmap["nodes"]
    edges = {i: list(v) for i, v in roadmap["edges"].items()}
    for q, idx in ((np.asarray(start, float), "start"), (np.asarray(goal, float), "goal")):
        new = len(nodes)
        order = np.argsort(np.linalg.norm(wrap(nodes - q), axis=1))[:k]
        nodes = np.vstack([nodes, q])
        edges[new] = []
        for j in order:
            j = int(j)
            free, _ = edge_free(q, nodes[j], obstacles, resolution)
            if free:
                d = distance(q, nodes[j])
                edges[new].append((j, d))
                edges[j].append((new, d))
        if idx == "start":
            s = new
        else:
            g = new

    dist = {s: 0.0}
    prev = {}
    heap = [(0.0, s)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == g:
            break
        if d > dist.get(u, np.inf):
            continue
        for v, w in edges[u]:
            nd = d + w
            if nd < dist.get(v, np.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    if g not in dist:
        return {"path": None, "seconds": time.perf_counter() - t0}
    path = [g]
    while path[-1] != s:
        path.append(prev[path[-1]])
    path.reverse()
    return {"path": [nodes[i] for i in path], "seconds": time.perf_counter() - t0}


def check_rate(n=20_000, seed=0):
    """Measure this machine's collision checks per second, so the grid table is not a guess."""
    rng = np.random.default_rng(seed)
    samples = rng.uniform(-np.pi, np.pi, (n, 2))
    t0 = time.perf_counter()
    for q in samples:
        collides(q)
    return n / (time.perf_counter() - t0)


def human_seconds(seconds):
    if seconds < 120:
        return f"{seconds:,.0f} s"
    if seconds < 86_400:
        return f"{seconds / 3600:,.1f} hours"
    if seconds < 86_400 * 730:
        return f"{seconds / 86_400:,.0f} days"
    return f"{seconds / (86_400 * 365):,.0f} years"


def grid_table():
    """What the exhaustive answer costs, per degree of freedom."""
    mine = check_rate()
    print(f"this machine checks {mine:,.0f} configurations per second")
    print("one pass over a grid at 5 degrees per joint, 72 cells per axis")
    print(f"  joints{'cells':>22}{'at this rate':>20}{'at 1e6 per second':>22}")
    for d in (2, 3, 6, 7):
        cells = 72.0 ** d
        print(f"  {d:>6}{cells:>22,.0f}"
              f"{human_seconds(cells / mine):>20}{human_seconds(cells / 1e6):>22}")
    print("  a 30-joint humanoid has 72**30 cells, which is about 10**55.")


def no_path():
    """Ask for a configuration that is reachable but not connected, and watch."""
    obstacles = BENCH + (EXTRA_DISC,)
    print(f"start {START} and goal {POCKET} are both collision-free: "
          f"{not collides(START, obstacles)} and {not collides(POCKET, obstacles)}")
    print("but the flood fill in the previous lesson put them in different components.")
    print("  budget    verdict     nodes   seconds")
    for budget in (500, 2_000, 8_000):
        out = rrt(START, POCKET, obstacles, seed=0, max_nodes=budget)
        verdict = "found" if out["path"] else "gave up"
        print(f"  {budget:>6,}   {verdict:>8}   {out['nodes']:>7,}   {out['seconds']:>7.2f}")
    print("The planner never says 'no path'. It only ever says 'not within this budget'.")


def audit(seeds=20):
    """Re-check finished paths at a resolution far finer than the planner used.

    Each row moves the extension length and the edge spacing together, which is
    what a real implementation does: a long step with a coarse check is fast and
    steps straight over thin obstacles.
    """
    print(f"paths accepted at a coarse edge check, re-checked at 0.005 rad, {seeds} seeds")
    print("  step  spacing   found   clean   worst path   seconds/run")
    for step, res in ((1.20, 1.20), (0.80, 0.80), (0.50, 0.50), (0.25, 0.25), (0.25, 0.05)):
        found = clean = worst = 0
        elapsed = 0.0
        for seed in range(seeds):
            out = rrt(START, GOAL, seed=seed, step=step, resolution=res)
            elapsed += out["seconds"]
            if out["path"] is None:
                continue
            found += 1
            bad = sum(not edge_free(out["path"][i], out["path"][i + 1], resolution=0.005)[0]
                      for i in range(len(out["path"]) - 1))
            clean += bad == 0
            worst = max(worst, bad)
        print(f"  {step:>4.2f}  {res:>7.2f}   {found:>5}   {clean:>5}   "
              f"{worst:>6} bad edges   {elapsed / seeds:>10.3f}")


def touch(target=BENCH[1], gaps=(0.20, 0.10, 0.05, 0.02, 0.00, -0.02)):
    """Ask the planner to put the fingertip a given gap from a disc's surface.

    The disc is an obstacle to the collision checker, so as the gap closes the
    goal configuration stops being legal and the planner stops answering. This is
    the boundary the next lesson is about: a planner's model has no vocabulary for
    deliberate contact.
    """
    from cspace_2link import L1, L2, elbow_and_tip

    ox, oy, orad = target
    reach = np.hypot(ox, oy)
    print(f"reaching for the disc at ({ox}, {oy}), radius {orad}, {reach:.2f} from the base")
    print("  gap to surface   goal legal?   planner")
    for gap in gaps:
        # Put the fingertip on the line from the base to the disc centre.
        want = reach - orad - gap
        # Elbow-up solution of the 2-link arm for a point at distance `want`
        # along the disc's bearing.
        cos2 = (want ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
        if abs(cos2) > 1.0:
            print(f"  {gap:>14.2f}   out of reach")
            continue
        t2 = float(np.arccos(cos2))
        bearing = float(np.arctan2(oy, ox))
        t1 = bearing - float(np.arctan2(L2 * np.sin(t2), L1 + L2 * np.cos(t2)))
        goal = np.array([t1, t2])
        legal = not collides(goal)
        if not legal:
            print(f"  {gap:>14.2f}   no            refuses: the goal is a collision")
            continue
        out = rrt(START, goal, seed=0)
        verdict = (f"solved in {out['seconds']:.2f} s, {out['nodes']} nodes"
                   if out["path"] else "gave up")
        print(f"  {gap:>14.2f}   yes           {verdict}")


def main():
    if "--grid" in sys.argv:
        return grid_table()
    if "--nopath" in sys.argv:
        return no_path()
    if "--audit" in sys.argv:
        return audit()
    if "--touch" in sys.argv:
        return touch()
    if "--prm" in sys.argv:
        road = prm()
        print(f"roadmap: 600 free samples, {road['kept']:,} of {road['tested']:,} candidate "
              f"edges kept, built in {road['seconds']:.2f} s")
        out = prm_query(road, START, GOAL)
        if out["path"] is None:
            print("  query failed: start or goal did not attach to the roadmap")
        else:
            print(f"  query answered in {1000 * out['seconds']:.0f} ms, "
                  f"path length {path_length(out['path']):.2f} rad")
        rng = np.random.default_rng(1)
        queries = []
        while len(queries) < 20:
            q = rng.uniform(-np.pi, np.pi, 2)
            if not collides(q):
                queries.append(q)
        t0 = time.perf_counter()
        hits = sum(prm_query(road, START, q)["path"] is not None for q in queries)
        print(f"  20 further queries against the same roadmap: {hits}/20 answered, "
              f"{time.perf_counter() - t0:.2f} s total")
        return

    out = rrt(START, GOAL)
    print(f"one RRT run, seed 0: {'solved' if out['path'] else 'failed'} in "
          f"{out['seconds']:.2f} s")
    print(f"  {out['samples']:,} samples, {out['nodes']:,} nodes kept, "
          f"{out['checks']:,} collision checks")
    print(f"  path: {len(out['path'])} waypoints, {path_length(out['path']):.2f} rad of travel")

    runs = [rrt(START, GOAL, seed=s) for s in range(40)]
    solved = [r for r in runs if r["path"] is not None]
    lengths = [path_length(r["path"]) for r in solved]
    print(f"40 seeds: {len(solved)}/40 solved, "
          f"median {np.median([r['seconds'] for r in solved]):.2f} s, "
          f"nodes {min(r['nodes'] for r in solved):,} to {max(r['nodes'] for r in solved):,}")
    print(f"  path length {min(lengths):.2f} to {max(lengths):.2f} rad "
          f"(median {np.median(lengths):.2f})")

    before = [path_length(r["path"]) for r in solved]
    after = [path_length(shortcut(r["path"], seed=i)) for i, r in enumerate(solved)]
    print(f"shortcutting: median {np.median(before):.2f} -> {np.median(after):.2f} rad "
          f"({100 * (1 - np.median(after) / np.median(before)):.0f}% shorter), "
          f"straight-line lower bound {distance(START, GOAL):.2f}")


if __name__ == "__main__":
    main()
