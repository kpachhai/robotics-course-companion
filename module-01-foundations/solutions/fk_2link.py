"""Solution - Lesson 1.11 FK (self-contained: local minimal transform helpers)."""
import numpy as np

L1, L2 = 1.0, 0.7


def _Rz4(a):
    c, s = np.cos(a), np.sin(a)
    T = np.eye(4); T[0, 0], T[0, 1], T[1, 0], T[1, 1] = c, -s, s, c
    return T


def _Tx4(d):
    T = np.eye(4); T[0, 3] = d
    return T


def fk_trig(t1, t2):
    x = L1 * np.cos(t1) + L2 * np.cos(t1 + t2)
    y = L1 * np.sin(t1) + L2 * np.sin(t1 + t2)
    return x, y, t1 + t2


def fk_transforms(t1, t2):
    T = _Rz4(t1) @ _Tx4(L1) @ _Rz4(t2) @ _Tx4(L2)
    x, y = T[0, 3], T[1, 3]
    phi = np.arctan2(T[1, 0], T[0, 0])
    return x, y, phi


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    for _ in range(10_000):
        t1, t2 = rng.uniform(-np.pi, np.pi, 2)
        a, b = np.array(fk_trig(t1, t2)), np.array(fk_transforms(t1, t2))
        da = (a[2] - b[2] + np.pi) % (2 * np.pi) - np.pi
        assert np.allclose(a[:2], b[:2], atol=1e-9) and abs(da) < 1e-9
    print("✅ solution FK cross-check passes.")
