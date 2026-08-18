"""Solution - Lesson 1.4 lab (the three matrices; everything else is unchanged)."""
import numpy as np


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])


# Memory aids: each leaves its own axis untouched (the 1 on the diagonal);
# the 2x2 cos/sin block lives in the other two rows/cols; Ry's minus sign sits
# bottom-left rather than top-right (cyclic order x->y->z keeps handedness).

if __name__ == "__main__":
    for M in (Rx(0.3), Ry(-1.1), Rz(2.0), Rz(1.0) @ Ry(0.5) @ Rx(-0.2)):
        assert np.allclose(M.T @ M, np.eye(3), atol=1e-12)
        assert abs(np.linalg.det(M) - 1) < 1e-12
    print("✅ solution matrices verified.")
