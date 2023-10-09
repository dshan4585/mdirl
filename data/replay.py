import tensorflow as tf
import numpy as np
from util.tf_util import *
from util.rms import rms, r_rms

np_split = np.split
np_ed = np.expand_dims
np_concat = np.concatenate
np_randint = np.random.randint
np_i32 = np.int32

def get_init_exploration(n, init_size, π, env):
  reset = env.reset
  step = env.step
  act = π.act

  s = reset()
  a = env.A.sample()

  states = np.empty((n,)+s.shape, dtype=np.float32)
  actions = np.empty((n,)+a.shape, dtype=np.float32)
  rewards = np.empty(n, dtype=np.float32)
  next_states = np.empty((n,)+s.shape, dtype=np.float32)
  absorbings = np.empty(n, np.bool)

  i = 0
  terminal = False

  while True:
    if i == init_size:
      return states, actions, next_states,\
             rewards, absorbings
    a = act(s)
    states[i] = s
    actions[i] = a
    ś, r, terminal, truncated = step(activ(a))
    absorbings[i] = terminal and not truncated
    next_states[i] = s = ś
    rewards[i] = r
    i += 1
    if terminal: s = reset()

@tf.function
def _get_next_batch(saś, b, r, indices, splits):
  return split(gather(saś, indices), splits, -1) +\
    [gather(b, indices), gather(r, indices)]

@tf.function
def _get_next_batch_adv(saś, indices, splits, dcount):
  return split(gather(saś, indices), splits, -1)[:dcount]

class ReplayBuffer:
  def __init__(self, n, init_size, π, env, ns, na, dintype=None):
    self.size = init_size
    self.n = n
    if dintype is not None:
      self.dcount = dintype.count('s') + dintype.count('a') + dintype.count('ś')
    s,a,ś,r,b = get_init_exploration(n, init_size, π, env)
    self.saś = np_concat((s,a,ś),-1)
    self.b = b
    self.r = r
    self.ptr = init_size
    self.splits = const([ns,na,ns],i32)

  def add(self, nb, τ):
    new_saś = np_concat((τ["s"], τ["a"], τ["ś"]),-1)
    new_b = τ["b"]
    new_r = τ["r_true"]

    next_ptr = self.ptr + nb
    if next_ptr < self.n:
      self.saś[self.ptr:next_ptr] = new_saś
      self.b[self.ptr:next_ptr] = new_b
      self.r[self.ptr:next_ptr] = new_r
      self.ptr = next_ptr
    else:
      if next_ptr == self.n:
        self.saś[self.ptr:] = new_saś
        self.b[self.ptr:] = new_b
        self.r[self.ptr:] = new_r
        self.ptr = 0
      else:
        nb2 = next_ptr - self.n
        nb1 = nb - nb2
        new_saś1, new_saś2 = np_split(new_saś, [nb1])
        new_b1, new_b2 = np_split(new_b, [nb1])
        new_r1, new_r2 = np_split(new_r, [nb1])

        self.saś[self.ptr:] = new_saś1
        self.saś[:nb2] = new_saś2
        self.b[self.ptr:] = new_b1
        self.b[:nb2] = new_b2
        self.r[self.ptr:] = new_r1
        self.r[:nb2] = new_r2
        self.ptr = nb2
    self.size = min(self.n, self.size + nb)

  def get_next_batch(self, batch_size):
    indices = np_randint(0, self.size, batch_size, dtype=np_i32)
    return _get_next_batch(self.saś, self.b, self.r, indices, self.splits)

  def get_next_batch_adv(self, batch_size):
    indices = np_randint(0, self.size, batch_size, dtype=np_i32)
    return _get_next_batch_adv(self.saś, indices, self.splits, self.dcount)

