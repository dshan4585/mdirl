import numpy as np
import gym

class Env:
  def __init__(self, name, render=False):
    self.name = name
    self.env = gym.make(name)
    self.S = self.env.observation_space
    self.A = self.env.action_space

  def step(self, a):
    ś, r, t, info = self.env.step(a)
    if t and 'TimeLimit.truncated' in info:
      truncated = info['TimeLimit.truncated']
    else:
      truncated = False
    return ś.astype('f4', copy=False), r.astype('f4', copy=False), t, truncated
  def reset(self):
    return self.env.reset().astype('f4', copy=False)
  def seed(self, i):
    self.env.seed(i)
  def close(self):
    self.env.close()

class AntEnv:
  def __init__(self, name, render=False):
    self.name = name
    self.env = gym.make(name)
    self.S = self.env.observation_space
    self.A = self.env.action_space

  def step(self, a):
    ś, r, t, info = self.env.step(a)
    if t and 'TimeLimit.truncated' in info:
      truncated = info['TimeLimit.truncated']
    else:
      truncated = False
    return ś[:27].astype('f4', copy=False), r.astype('f4', copy=False), t, truncated
  def reset(self):
    return self.env.reset()[:27].astype('f4', copy=False)
  def seed(self, i):
    self.env.seed(i)
  def close(self):
    self.env.close()
