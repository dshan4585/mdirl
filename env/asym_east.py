import numpy as np
import gym
from gym.spaces import Box
from gym.utils import seeding
from numpy.linalg import norm

class AsymmetricEastEnv(gym.Env):
  def __init__(self):
    self.threshold = 1.0
    self.r_coeff = 20.
    self.g_coeff = 5.
    self.c_coeff = 1e-2
    self.live_cost = 0.
    self.action_space = Box(-1, 1, (2,), np.float32)
    self.observation_space = Box(
        np.asarray([-9,-9]),
        np.asarray([14,9]), (2,), dtype=np.float32)
    self.p = np.array([0, 0], np.float32)
    self.gs = np.array([
        [12,0], [0,6],
        [-6,0], [0,-6]], np.float32)
    self.idx = 0
    self.info = {}
    self.reset()

  def step(self, a):
    clipped_a = np.clip(a, -1, 1)
    p_prev = self.p
    self.p = np.clip(self.p + clipped_a, [-9,-9], [14,9])

    pot = self.potential
    potential_diff = pot - self.potential_prev

    distance_r = self.r_coeff * potential_diff
    act_sq = np.square(a)
    ctrl_cost = 0.5 * np.sum(act_sq) * self.c_coeff
    is_goal = self.is_goal
    goal_bonus = self.g_coeff if is_goal else 0.
    r = distance_r + goal_bonus - ctrl_cost - self.live_cost
    self.potential_prev = pot
    return self.p, r, is_goal, self.info

  @property
  def is_goal(self):
    return self.distance <= self.threshold

  def renew_idx(self):
    self.idx = np.argmin(norm(self.p[None] - self.gs, axis=1))
    return self.idx

  @property
  def potential(self):
    d = norm(self.p - self.gs[self.idx])
    gau = np.exp(-.5 * np.square(d/1.8))
    return gau

  @property
  def distance(self):
    idx = self.idx
    g = self.gs[idx]
    d = norm(self.p - g)
    return d

  def reset(self):
    self.p = np.random.normal(scale=0.1, size=2)
    self.potential_prev = self.potential
    return self.p

  def seed(self, seed):
    pass

  def close(self):
    pass