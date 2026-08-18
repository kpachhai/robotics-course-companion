"""Lesson 3.15 - the plant. Complete; nothing here is yours to write.

A batch of independent cart-poles stepped together with numpy, so the policy
network sees one batched forward pass per timestep instead of one per
environment. That is the same trick GPU sim uses, four thousand times smaller.

State is (x, x_dot, theta, theta_dot). Two actions: push left, push right.
Reward is 1 per step the pole is still up. An episode ends when the cart leaves
the track or the pole passes 12 degrees, and is truncated at MAX_T steps.
"""
import math

import numpy as np

GRAV, M_CART, M_POLE, HALF_LEN, FORCE, TAU = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
M_TOTAL, POLE_ML = M_CART + M_POLE, M_POLE * HALF_LEN
X_LIMIT, TH_LIMIT, MAX_T = 2.4, 12 * math.pi / 180, 200


class VecCartPole:
    """`n` cart-poles in lockstep. No auto-reset: an episode that ends stays
    ended, and its rows are masked out of the batch."""

    def __init__(self, n, rng):
        self.n, self.rng = n, rng

    def reset(self):
        self.state = self.rng.uniform(-0.05, 0.05, size=(self.n, 4))
        self.alive = np.ones(self.n, dtype=bool)
        return self.state.copy()

    def step(self, action):
        x, xd, th, thd = self.state.T
        force = np.where(action == 1, FORCE, -FORCE)
        cos_th, sin_th = np.cos(th), np.sin(th)
        temp = (force + POLE_ML * thd * thd * sin_th) / M_TOTAL
        thdd = ((GRAV * sin_th - cos_th * temp)
                / (HALF_LEN * (4.0 / 3.0 - M_POLE * cos_th * cos_th / M_TOTAL)))
        xdd = temp - POLE_ML * thdd * cos_th / M_TOTAL

        live = self.alive                      # frozen rows integrate nothing
        x = x + TAU * xd * live
        xd = xd + TAU * xdd * live
        th = th + TAU * thd * live
        thd = thd + TAU * thdd * live
        self.state = np.stack([x, xd, th, thd], axis=1)
        self.alive = self.alive & (np.abs(x) <= X_LIMIT) & (np.abs(th) <= TH_LIMIT)
        return self.state.copy(), live.astype(np.float32), self.alive.copy()
