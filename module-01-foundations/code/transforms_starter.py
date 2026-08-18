"""Lessons 1.8-1.9 - build transforms.py, YOUR library for the rest of the course.

Fill each TODO, then:  python transforms_starter.py --test
When green, this file (renamed transforms.py in your head) is a dependency of
lessons 1.11-1.19. Keep it clean.
"""
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R


def make_T(Rm, t):
    """4x4 homogeneous transform from 3x3 rotation Rm and length-3 translation t."""
    # TODO(you): build the block matrix [[R, t], [0 0 0 1]].
    raise NotImplementedError


def apply_T(T, p):
    """Apply transform T to 3D point p (returns a 3-vector)."""
    # TODO(you): homogeneous form -> multiply -> back to 3D.
    raise NotImplementedError


def compose(*Ts):
    """compose(T_AB, T_BC, T_CD) -> T_AD. Left-to-right reads outer-to-inner."""
    # TODO(you): fold with matrix multiplication. Mind the order.
    raise NotImplementedError


def inv_T(T):
    """Closed-form inverse: [[Rᵀ, −Rᵀ t], [0, 1]]. Do NOT call np.linalg.inv."""
    # TODO(you)
    raise NotImplementedError


def from_quat_trans(q_xyzw, t):
    """Pose from quaternion [x,y,z,w] + translation (the URDF/MJCF-adjacent form)."""
    # TODO(you): scipy R.from_quat gives the 3x3; reuse make_T.
    raise NotImplementedError


def to_quat_trans(T):
    """Inverse of from_quat_trans -> (q_xyzw, t)."""
    # TODO(you)
    raise NotImplementedError


# ---------------------------------------------------------------- tests ----
def _random_T(rng):
    return make_T(R.random(rng=rng).as_matrix(), rng.uniform(-2, 2, 3))


def run_tests(n=1000):
    rng = np.random.default_rng(42)
    I = np.eye(4)
    for _ in range(n):
        A, B, C = _random_T(rng), _random_T(rng), _random_T(rng)
        p = rng.uniform(-2, 2, 3)

        # 1) inverse round-trips
        assert np.allclose(compose(A, inv_T(A)), I, atol=1e-9)
        assert np.allclose(compose(inv_T(A), A), I, atol=1e-9)
        # 2) associativity
        assert np.allclose(compose(compose(A, B), C), compose(A, compose(B, C)), atol=1e-9)
        # 3) apply matches compose: (A∘B)(p) == A(B(p))
        assert np.allclose(apply_T(compose(A, B), p), apply_T(A, apply_T(B, p)), atol=1e-9)
        # 4) rotation block stays orthonormal through composition
        Rm = compose(A, B, C)[:3, :3]
        assert np.allclose(Rm.T @ Rm, np.eye(3), atol=1e-9)
        # 5) quat round trip
        q, t = to_quat_trans(A)
        assert np.allclose(from_quat_trans(q, t), A, atol=1e-9)
    print(f"✅ transforms library: all property tests passed over {n} random poses.")

    # 6) one concrete, hand-checkable case (the lesson's -Rᵀt point):
    T90 = make_T(R.from_euler("z", 90, degrees=True).as_matrix(), [1, 0, 0])
    assert np.allclose(apply_T(inv_T(T90), apply_T(T90, [0.2, 0.3, 0.0])), [0.2, 0.3, 0.0], atol=1e-12)
    wrong_inv = make_T(T90[:3, :3].T, -T90[:3, 3])   # the -t (no Rᵀ) mistake
    assert not np.allclose(wrong_inv, inv_T(T90))
    print("✅ and the −Rᵀt subtlety is demonstrably not −t.")


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_tests()
    else:
        print("Implement the TODOs, then run:  python transforms_starter.py --test")
