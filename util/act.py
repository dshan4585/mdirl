import numpy as np
import tensorflow as tf
from util.tf_util import *

def generate_τ(π, env, size):
  reset = env.reset
  step = env.step
  act = π.act

  s = reset()
  a = env.A.sample()
  states = np.empty((size,)+s.shape, dtype=np.float32)
  actions = np.empty((size,)+a.shape, dtype=np.float32)
  rewards = np.empty(size, dtype=np.float32)
  next_states = np.empty((size,)+s.shape, dtype=np.float32)
  terminals = np.empty(size, np.bool)
  absorbings = np.empty(size, np.bool)
  initials = np.empty(size, np.bool)

  cnt = 0
  new = True
  cur_len = 0
  cur_ret = np.float32(0.)
  ep_len = []
  ep_ret = []

  terminal = False

  while True:
    if cnt > 0 and cnt % size == 0:
      yield {
          "s": states,
          "a": actions,
          "r_true": rewards,
          "ś": next_states,
          "t": terminals,
          "b": absorbings,
          "i": initials,
          "ep_len": ep_len,
          "ep_ret": ep_ret}
      ep_ret = []
      ep_len = []

    i = cnt % size

    initials[i] = new
    a = act(s)
    states[i] = s
    actions[i] = a
    ś, r, terminal, truncated = step(activ(a))
    terminals[i] = terminal
    absorbings[i] = terminal and not truncated
    new = terminal
    rewards[i] = r
    next_states[i] = s = ś
    cur_ret += r
    cur_len += 1
    cnt += 1
    if terminal:
      ep_len.append(cur_len)
      ep_ret.append(cur_ret)
      cur_ret = np.float32(0.)
      cur_len = 0
      s = reset()

def generate_τ2(π, env, size):
  reset = env.reset
  step = env.step
  act = π.act

  s = reset()
  rewards = np.empty(size, dtype=np.float32)

  cnt = 0
  new = True
  cur_len = 0
  cur_ret = np.float32(0.)
  ep_len = []
  ep_ret = []

  terminal = False

  while True:
    if cnt > 0 and cnt % size == 0:
      s = reset()
      yield {
          "r_true": rewards.copy(),
          "ep_len": ep_len,
          "ep_ret": ep_ret}
      ep_ret = []
      ep_len = []

    i = cnt % size

    a = act(s)
    ś, r, terminal, truncated = step(activ(a))
    new = terminal
    s = ś
    cur_ret += r
    cur_len += 1
    cnt += 1
    if terminal:
      ep_len.append(cur_len)
      ep_ret.append(cur_ret)
      cur_ret = np.float32(0.)
      cur_len = 0
      s = reset()
