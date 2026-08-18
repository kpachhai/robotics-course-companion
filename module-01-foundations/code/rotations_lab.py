"""Lessons 1.4-1.7 lab - rotation matrices, non-commutativity, double cover, slerp.

Run:  python rotations_lab.py        (tests + saves slerp_vs_euler.png)
"""
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


def Rx(a):
    # TODO(you): 3x3 rotation about x by angle a (radians).
    raise NotImplementedError

def Ry(a):
    # TODO(you)
    raise NotImplementedError

def Rz(a):
    # TODO(you)
    raise NotImplementedError


def is_rotation(M, tol=1e-9):
    """A debugging tool you'll reuse forever: is M a valid rotation matrix?"""
    return (np.allclose(M.T @ M, np.eye(3), atol=tol)
            and abs(np.linalg.det(M) - 1.0) < tol)


def part1_properties():
    rng = np.random.default_rng(0)
    for _ in range(100):
        M = Rz(rng.uniform(-np.pi, np.pi)) @ Ry(rng.uniform(-np.pi, np.pi)) @ Rx(rng.uniform(-np.pi, np.pi))
        assert is_rotation(M)
    # Cross-check against scipy's convention:
    a = 0.7
    assert np.allclose(Rz(a), R.from_euler("z", a).as_matrix(), atol=1e-12)
    print("✅ part 1: Rx/Ry/Rz valid rotations, matching scipy")


def part2_order_matters():
    p = np.array([1.0, 0.0, 0.0])
    a = np.pi / 2
    p_zx = Rz(a) @ Rx(a) @ p        # apply Rx FIRST (rightmost acts first)
    p_xz = Rx(a) @ Rz(a) @ p
    print(f"   Rz∘Rx applied to x̂ -> {np.round(p_zx, 3)}")
    print(f"   Rx∘Rz applied to x̂ -> {np.round(p_xz, 3)}")
    assert not np.allclose(p_zx, p_xz)
    print("✅ part 2: 3D rotation order matters (now do it with your phone)")


def part3_double_cover():
    q = R.from_euler("xyz", [40, 20, 10], degrees=True).as_quat()  # [x,y,z,w]
    q_neg = -q
    v = np.array([0.3, -0.5, 0.8])
    v1 = R.from_quat(q).apply(v)
    v2 = R.from_quat(q_neg).apply(v)
    assert np.allclose(v1, v2, atol=1e-12)
    mse = float(np.mean((q - q_neg) ** 2))
    print(f"✅ part 3: q and −q rotate identically, yet MSE(q,−q) = {mse:.3f} "
          f"- the loss-function trap from the lesson")


def part4_slerp_vs_euler(save=None):
    """Interpolate between two orientations two ways; plot the path of a test vector.

    `save` defaults to a path beside this script rather than the current
    directory: run from the repo root, a bare filename would drop an untracked
    PNG there, and only `module-*/code/*.png` is gitignored.
    """
    save = Path(__file__).with_name("slerp_vs_euler.png") if save is None else Path(save)
    e0, e1 = np.array([0.0, 0.0, 0.0]), np.array([170.0, 60.0, 20.0])  # degrees, xyz
    r0, r1 = R.from_euler("xyz", e0, degrees=True), R.from_euler("xyz", e1, degrees=True)
    ts = np.linspace(0, 1, 60)
    v = np.array([1.0, 0.0, 0.0])

    euler_path = np.array([R.from_euler("xyz", e0 + t * (e1 - e0), degrees=True).apply(v) for t in ts])
    slerp_path = Slerp([0, 1], R.concatenate([r0, r1]))(ts).apply(v)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(slerp_path[:, 0], slerp_path[:, 1], lw=2, color="#2a78d6", label="slerp (quaternion)")
    ax.plot(euler_path[:, 0], euler_path[:, 1], lw=2, color="#eb6834", label="euler lerp")
    ax.scatter(*slerp_path[0, :2], color="#0b0b0b", zorder=3)
    ax.annotate("start", slerp_path[0, :2], textcoords="offset points", xytext=(6, 6), color="#52514e")
    ax.set_aspect("equal"); ax.legend(frameon=False)
    ax.set_title("Path of a rotated vector: slerp arcs cleanly; euler lerp detours")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.savefig(save, dpi=150, bbox_inches="tight")
    print(f"✅ part 4: wrote {save.name} - look at the orange detour")


if __name__ == "__main__":
    part1_properties()
    part2_order_matters()
    part3_double_cover()
    part4_slerp_vs_euler()
