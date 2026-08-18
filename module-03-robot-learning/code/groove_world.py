"""A world you can actually act in. Given to you complete.

Lessons 3.4 and 3.5 need something the SO-101 dataset cannot provide: a robot
you can *run a policy on*. A recording has no opinion about what happens when
the policy makes a mistake, because in a recording the policy never acts.

So this is the smallest world that still breaks the same way a real one does.
It is the 2-link planar arm from Module 1, dragging a pin along a curved groove
milled into a plate:

    observation   (theta1, theta2)          the two joint angles, radians
    action        (dtheta1, dtheta2)        how far to turn each joint, radians
    failure       the pin leaves the groove

Two dimensions in, two out, so the training data and the policy's own
trajectory can be drawn on the same sheet of paper. That is the entire reason
for the small numbers - everything here scales to six joints and a camera, and
none of it gets easier when it does.

Run:  python groove_world.py         check the scripted expert
"""
import numpy as np

L1, L2 = 1.0, 0.7          # link lengths, same arm as Module 1
DEMO_STEPS = 60            # how long a demonstration runs
EVAL_STEPS = 240           # how long we let a policy run when testing it
BAND = 0.05                # groove half-width; leave it and the pin jams
REACH = 0.04               # "arrived at B" radius
GAIN = 0.6                 # expert: fraction of the error closed per step
LOOK = 14                  # expert lookahead, in path samples
LAM = 0.05                 # damped-least-squares damping (Module 1, lesson 15)
MAX_STEP = 0.15            # rad, per-step cap on how far a joint may turn
HOME = np.array([0.20, 1.35])
JITTER = 0.02              # rad, how much the start pose varies between demos


def fk(theta):
    """Joint angles to pin position. Forward kinematics, Module 1 lesson 11."""
    t1, t2 = theta[..., 0], theta[..., 1]
    return np.stack([L1 * np.cos(t1) + L2 * np.cos(t1 + t2),
                     L1 * np.sin(t1) + L2 * np.sin(t1 + t2)], axis=-1)


def jacobian(theta):
    """How pin position changes per radian of each joint. Module 1 lesson 13."""
    t1, t2 = theta[0], theta[1]
    s1, c1 = np.sin(t1), np.cos(t1)
    s12, c12 = np.sin(t1 + t2), np.cos(t1 + t2)
    return np.array([[-L1 * s1 - L2 * s12, -L2 * s12],
                     [L1 * c1 + L2 * c12,   L2 * c12]])


def dls(theta, move):
    """Joint change that moves the pin by `move`. Damped least squares."""
    J = jacobian(theta)
    return J.T @ np.linalg.solve(J @ J.T + LAM ** 2 * np.eye(2), move)


A_TIP = fk(HOME)                        # where the groove starts
B_TIP = np.array([1.15, 0.35])          # where it ends
_BOW = np.array([0.35, 0.30])           # control point: the groove is curved
_u = np.linspace(0, 1, 400)[:, None]
PATH = (1 - _u) ** 2 * A_TIP + 2 * (1 - _u) * _u * _BOW + _u ** 2 * B_TIP


def nearest_on_path(point):
    """(index of the closest groove sample, distance to it)."""
    d = np.linalg.norm(PATH - point, axis=1)
    i = int(d.argmin())
    return i, float(d[i])


def expert(theta):
    """The scripted demonstrator: look a little way along the groove, go there.

    This is pure pursuit - the trajectory tracker from Module 1 lesson 18. It is
    a rule, not a recording, so it is correct at every pose in the plane, including
    poses no demonstration ever visited. That is exactly what the clone will not be.
    """
    pin = fk(theta)
    i, _ = nearest_on_path(pin)
    target = PATH[min(i + LOOK, len(PATH) - 1)]
    step = GAIN * dls(theta, target - pin)
    size = np.linalg.norm(step)
    return step * MAX_STEP / size if size > MAX_STEP else step


def start_pose(rng, offset=0.0):
    """Home, jittered like a real demonstration, then pushed `offset` off centre."""
    theta = HOME + rng.normal(0, JITTER, 2)
    if offset:
        tangent = PATH[6] - PATH[0]
        tangent = tangent / np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]])
        sign = 1.0 if rng.random() < 0.5 else -1.0
        theta = theta + dls(theta, sign * offset * normal)
    return theta


def rollout(policy, theta0, steps=EVAL_STEPS, walls=True):
    """Run `policy` in closed loop. Returns everything the lessons measure."""
    theta = np.array(theta0, dtype=float)
    states, actions, pins, devs = [], [], [], []
    crash_step, reached = None, False
    for t in range(steps):
        action = np.asarray(policy(theta), dtype=float)
        if not np.all(np.isfinite(action)):
            crash_step = t
            break
        states.append(theta.copy())
        actions.append(action)
        theta = theta + action
        pin = fk(theta)
        dev = nearest_on_path(pin)[1]
        pins.append(pin)
        devs.append(dev)
        reached = reached or np.linalg.norm(pin - B_TIP) < REACH
        if walls and t >= 3 and dev > BAND:
            crash_step = t + 1
            break
    return dict(
        states=np.array(states), actions=np.array(actions),
        pins=np.array(pins), devs=np.array(devs),
        crashed=crash_step is not None,
        survived=steps if crash_step is None else crash_step,
        reached_b=bool(reached),
        success=bool(reached and crash_step is None),
    )


def collect(n_episodes, seed, noise=0.0, recovery=0.0):
    """Demonstrate the task `n_episodes` times, keeping every (state, action) pair.

    noise     std of the disturbance added to what actually gets executed, so the
              expert is forced to demonstrate corrections (lesson 3.5)
    recovery  half the episodes start this far off centre (lesson 3.5)
    """
    rng = np.random.default_rng(seed)
    states, actions = [], []
    for k in range(n_episodes):
        offset = recovery * rng.uniform(0.3, 1.0) if (recovery and k % 2) else 0.0
        theta = start_pose(rng, offset)
        for _ in range(DEMO_STEPS):
            action = expert(theta)
            states.append(theta.copy())
            actions.append(action)
            theta = theta + (action + rng.normal(0, noise, 2) if noise else action)
    return np.array(states), np.array(actions)


def fit_policy(states, actions, epochs=200, seed=0, hidden=128):
    """Behaviour cloning, exactly the recipe from lesson 3.3, packaged.

    Standardise both sides, fit a small MLP with mean-squared error, hand back a
    function with the same signature as `expert`. Returns (policy, history).
    """
    import torch
    from torch import nn

    torch.manual_seed(seed)
    x_mean, x_std = states.mean(0), states.std(0) + 1e-8
    y_mean, y_std = actions.mean(0), actions.std(0) + 1e-8
    X = torch.tensor((states - x_mean) / x_std, dtype=torch.float32)
    Y = torch.tensor((actions - y_mean) / y_std, dtype=torch.float32)

    gen = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(X), generator=gen)
    cut = int(0.9 * len(X))
    train_idx, val_idx = order[:cut], order[cut:]

    net = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(),
                        nn.Linear(hidden, hidden), nn.ReLU(),
                        nn.Linear(hidden, 2))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    history = []
    for _ in range(epochs):
        batch_order = train_idx[torch.randperm(len(train_idx), generator=gen)]
        running = 0.0
        for i in range(0, len(batch_order), 256):
            b = batch_order[i:i + 256]
            loss = ((net(X[b]) - Y[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * len(b)
        with torch.no_grad():
            val = ((net(X[val_idx]) - Y[val_idx]) ** 2).mean().item()
        history.append((running / len(batch_order), val))

    def policy(theta):
        with torch.no_grad():
            x = torch.tensor((theta - x_mean) / x_std, dtype=torch.float32)
            return net(x).numpy() * y_std + y_mean

    return policy, np.array(history)


def evaluate(policy, n=40, seed=7, offset=0.0, steps=EVAL_STEPS):
    """Success rate and median steps survived, over `n` fresh episodes."""
    rng = np.random.default_rng(seed)
    runs = [rollout(policy, start_pose(rng, offset), steps=steps) for _ in range(n)]
    return dict(
        success=sum(r["success"] for r in runs) / n,
        reached=sum(r["reached_b"] for r in runs) / n,
        survived=float(np.median([r["survived"] for r in runs])),
        runs=runs,
    )


if __name__ == "__main__":
    print(f"groove: A {A_TIP.round(3)} -> B {B_TIP.round(3)}, "
          f"half-width {BAND}, demo {DEMO_STEPS} steps, eval {EVAL_STEPS} steps")
    for offset in (0.0, 0.02, 0.04):
        r = evaluate(expert, offset=offset)
        print(f"  expert, started {offset * 100:.0f} cm off centre: "
              f"success {r['success']:.2f}, survived {r['survived']:.0f}/{EVAL_STEPS}")
    print("The expert is a rule. It is right everywhere, so it never runs out of road.")
