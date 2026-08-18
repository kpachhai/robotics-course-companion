"""Lessons 1.13-1.15 - Jacobians and damped-least-squares IK for the 2-link arm.

Run:  python ik_2link.py                 (jacobian checks + 100-target IK eval)
      python ik_2link.py --singularity   (λ=0 vs λ=0.01 near the workspace edge)
"""
import sys
import numpy as np

L1, L2 = 1.0, 0.7


def fk(t1, t2):
    return np.array([L1 * np.cos(t1) + L2 * np.cos(t1 + t2),
                     L1 * np.sin(t1) + L2 * np.sin(t1 + t2)])


def jacobian_analytic(t1, t2):
    """2x2: rows = (x, y), cols = (θ1, θ2). From the lesson's formula."""
    # TODO(you)
    raise NotImplementedError


def jacobian_numeric(t1, t2, eps=1e-6):
    """Finite differences: nudge each joint by eps, watch the end-effector."""
    # TODO(you): columns are (fk(θ + eps·e_j) − fk(θ)) / eps
    raise NotImplementedError


def ik_dls(target, theta0, lam=0.01, alpha=0.5, iters=200, tol=1e-6):
    """Damped least squares. Returns (theta, converged, n_iters, path)."""
    th = np.array(theta0, dtype=float)
    path = [th.copy()]
    for i in range(iters):
        e = target - fk(*th)
        if np.linalg.norm(e) < tol:
            return th, True, i, np.array(path)
        # TODO(you): J = jacobian_analytic(*th)
        #            dth = Jᵀ (J Jᵀ + λ² I)⁻¹ e     (2x2 inverse; np.linalg.solve is fine)
        #            th += alpha * dth ; path.append(th.copy())
        raise NotImplementedError
    return th, False, iters, np.array(path)


def check_jacobians(n=1000):
    rng = np.random.default_rng(3)
    for _ in range(n):
        t1, t2 = rng.uniform(-np.pi, np.pi, 2)
        assert np.allclose(jacobian_analytic(t1, t2), jacobian_numeric(t1, t2), atol=1e-4)
    print(f"✅ analytic and finite-difference Jacobians agree ({n} configs).")
    d = np.linalg.det(jacobian_analytic(0.3, 0.0))
    print(f"   det J at θ2=0 (arm straight): {d:.2e}  ← the singularity, on cue")


def eval_ik(n=100):
    rng = np.random.default_rng(4)
    ok, iters = 0, []
    for _ in range(n):
        r = rng.uniform(abs(L1 - L2) + 0.03, L1 + L2 - 0.03)   # reachable, off the edges
        a = rng.uniform(-np.pi, np.pi)
        target = r * np.array([np.cos(a), np.sin(a)])
        th, conv, k, _ = ik_dls(target, theta0=(0.4, 0.6))
        err = np.linalg.norm(fk(*th) - target)
        ok += conv and err < 1e-5
        iters.append(k)
    print(f"✅ IK: {ok}/{n} targets converged; median iterations {int(np.median(iters))}.")


def singularity_experiment():
    target = np.array([L1 + L2 - 1e-3, 0.0])     # a hair inside the outer edge
    for lam in (0.0, 0.01):
        th = np.array([0.3, 0.5])
        max_step = 0.0
        for _ in range(100):
            e = target - fk(*th)
            J = jacobian_analytic(*th)
            JJt = J @ J.T + (lam ** 2) * np.eye(2)
            dth = J.T @ np.linalg.solve(JJt, e)
            max_step = max(max_step, np.linalg.norm(dth))
            th += 0.5 * dth
        print(f"   λ={lam:<5} max ‖Δθ‖ per step = {max_step:8.2f}   "
              f"final error = {np.linalg.norm(fk(*th) - target):.2e}")
    print("   ← λ=0 whips the joints near the singular edge; damping stays calm.")


if __name__ == "__main__":
    if "--singularity" in sys.argv:
        singularity_experiment()
    else:
        check_jacobians()
        eval_ik()
