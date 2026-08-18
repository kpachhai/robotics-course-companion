"""Lesson 1.2 warm-up - points in two frames, by hand.

Frame {B} sits in the world {W}: translated by t = (1, 2), rotated +30° CCW.
Use only math.cos/math.sin; the point is to feel the operations once.

Run:  python frames_warmup.py     (asserts tell you when you're right)
"""
import math

THETA = math.radians(30.0)     # rotation of {B} relative to {W}
T = (1.0, 2.0)                 # origin of {B}, expressed in {W}


def rotate(p, theta):
    """Rotate 2D point p = (x, y) counterclockwise by theta. No translation."""
    x, y = p
    # TODO(you): return the rotated (x', y') using cos/sin.
    # Columns rule: x-axis lands on (cosθ, sinθ), y-axis on (−sinθ, cosθ).
    raise NotImplementedError


def b_to_w(p_B):
    """A point described in {B} -> its description in {W}.  Rotate THEN translate."""
    # TODO(you): use rotate() and T.
    raise NotImplementedError


def w_to_b(p_W):
    """A point described in {W} -> its description in {B}.  Undo b_to_w.
    Hint: subtract first, then rotate by -THETA. (Why this order? Lesson 1.8
    gives the general answer; here, reason it out: you're inverting
    'rotate then translate'.)"""
    # TODO(you)
    raise NotImplementedError


def close(a, b, tol=1e-9):
    return abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol


if __name__ == "__main__":
    # 1. A point one unit along {B}'s x-axis. Where is it in the world?
    p_W = b_to_w((1.0, 0.0))
    assert close(p_W, (1.0 + math.cos(THETA), 2.0 + math.sin(THETA))), p_W

    # 2. {B}'s own origin, described in {B}, is (0,0). In the world it must be T.
    assert close(b_to_w((0.0, 0.0)), T)

    # 3. Round trip: any point must survive W->B->W.
    p = (0.7, -1.3)
    assert close(b_to_w(w_to_b(p)), p)

    # 4. The classic mistake, demonstrated: translate-then-rotate is NOT b_to_w.
    wrong = rotate((1.0 + T[0], 0.0 + T[1]), THETA)
    assert not close(wrong, b_to_w((1.0, 0.0))), "order shouldn't matter?! recheck"

    print("✅ all frame conversions correct - you felt the order matter.")
