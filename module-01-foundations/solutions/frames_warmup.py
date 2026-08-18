"""Solution - Lesson 1.2 warm-up. Read only after your version passes (or truly stalls)."""
import math

THETA = math.radians(30.0)
T = (1.0, 2.0)


def rotate(p, theta):
    x, y = p
    c, s = math.cos(theta), math.sin(theta)
    return (c * x - s * y, s * x + c * y)


def b_to_w(p_B):
    rx, ry = rotate(p_B, THETA)          # 1) express in world axis directions
    return (rx + T[0], ry + T[1])        # 2) shift by where B's origin sits


def w_to_b(p_W):
    # Invert 'rotate then translate': first un-shift, then un-rotate.
    dx, dy = p_W[0] - T[0], p_W[1] - T[1]
    return rotate((dx, dy), -THETA)


def close(a, b, tol=1e-9):
    return abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol


if __name__ == "__main__":
    p_W = b_to_w((1.0, 0.0))
    assert close(p_W, (1.0 + math.cos(THETA), 2.0 + math.sin(THETA)))
    assert close(b_to_w((0.0, 0.0)), T)
    p = (0.7, -1.3)
    assert close(b_to_w(w_to_b(p)), p)
    wrong = rotate((1.0 + T[0], 0.0 + T[1]), THETA)
    assert not close(wrong, b_to_w((1.0, 0.0)))
    print("✅ solution self-check passes.")
