"""Solution - the transforms library (Lessons 1.8-1.9). Also importable by later solutions."""
import numpy as np
from scipy.spatial.transform import Rotation as R


def make_T(Rm, t):
    T = np.eye(4)
    T[:3, :3] = np.asarray(Rm, dtype=float)
    T[:3, 3] = np.asarray(t, dtype=float)
    return T


def apply_T(T, p):
    ph = np.append(np.asarray(p, dtype=float), 1.0)
    return (T @ ph)[:3]


def compose(*Ts):
    out = np.eye(4)
    for T in Ts:
        out = out @ T
    return out


def inv_T(T):
    Rm, t = T[:3, :3], T[:3, 3]
    return make_T(Rm.T, -Rm.T @ t)


def from_quat_trans(q_xyzw, t):
    return make_T(R.from_quat(q_xyzw).as_matrix(), t)


def to_quat_trans(T):
    return R.from_matrix(T[:3, :3]).as_quat(), T[:3, 3].copy()


if __name__ == "__main__":
    # Reuse the starter's test suite against this implementation.
    import importlib.util, pathlib, sys
    spec = importlib.util.spec_from_file_location(
        "starter", pathlib.Path(__file__).resolve().parents[0].parent / "code" / "transforms_starter.py")
    starter = importlib.util.module_from_spec(spec)
    sys.modules["starter"] = starter
    spec.loader.exec_module(starter)
    for name in ("make_T", "apply_T", "compose", "inv_T", "from_quat_trans", "to_quat_trans"):
        setattr(starter, name, globals()[name])
    starter.run_tests()
