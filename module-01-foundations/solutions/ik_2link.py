"""Solution - Lessons 1.13-1.15 (Jacobian + DLS IK), self-contained."""
import numpy as np

L1, L2 = 1.0, 0.7


def fk(t1, t2):
    return np.array([L1 * np.cos(t1) + L2 * np.cos(t1 + t2),
                     L1 * np.sin(t1) + L2 * np.sin(t1 + t2)])


def jacobian_analytic(t1, t2):
    s1, c1 = np.sin(t1), np.cos(t1)
    s12, c12 = np.sin(t1 + t2), np.cos(t1 + t2)
    return np.array([[-L1 * s1 - L2 * s12, -L2 * s12],
                     [ L1 * c1 + L2 * c12,  L2 * c12]])


def jacobian_numeric(t1, t2, eps=1e-6):
    base = fk(t1, t2)
    return np.column_stack([(fk(t1 + eps, t2) - base) / eps,
                            (fk(t1, t2 + eps) - base) / eps])


def ik_dls(target, theta0, lam=0.01, alpha=0.5, iters=200, tol=1e-6):
    th = np.array(theta0, dtype=float)
    path = [th.copy()]
    for i in range(iters):
        e = target - fk(*th)
        if np.linalg.norm(e) < tol:
            return th, True, i, np.array(path)
        J = jacobian_analytic(*th)
        dth = J.T @ np.linalg.solve(J @ J.T + (lam ** 2) * np.eye(2), e)
        th += alpha * dth
        path.append(th.copy())
    return th, False, iters, np.array(path)


def ik_analytic(target, elbow_up=True):
    """Law-of-cosines closed form; the 'bonus round' from the lesson."""
    x, y = target
    d2 = x * x + y * y
    c2 = (d2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if abs(c2) > 1.0:
        return None                       # unreachable
    t2 = np.arccos(np.clip(c2, -1, 1)) * (1 if elbow_up else -1)
    t1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(t2), L1 + L2 * np.cos(t2))
    return np.array([t1, t2])


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    for _ in range(1000):
        t = rng.uniform(-np.pi, np.pi, 2)
        assert np.allclose(jacobian_analytic(*t), jacobian_numeric(*t), atol=1e-4)
    for _ in range(200):
        r = rng.uniform(abs(L1 - L2) + 0.03, L1 + L2 - 0.03)
        a = rng.uniform(-np.pi, np.pi)
        target = r * np.array([np.cos(a), np.sin(a)])
        th, conv, _, _ = ik_dls(target, (0.4, 0.6))
        assert conv and np.linalg.norm(fk(*th) - target) < 1e-5
        an = ik_analytic(target)          # matches one branch:
        assert an is None or np.linalg.norm(fk(*an) - target) < 1e-9
    print("✅ solution IK: DLS converges on 200/200; analytic branch verified.")
