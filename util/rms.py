import numpy as np
import tensorflow as tf
from const import ns
from util.tf_util import *

@tf.function
def compute_vec(x):
  return concat([rsum(x,0),rsum(sq(x),0),ed(cast(shape(x)[0],f32),0)],0)

class RMS(object):
  def __init__(self, n, decay=0.999):
    self.n = n
    self._sum = var(zeros(n, f32))
    self._sumsq = var(ones(n, f32))
    self._cnt = var(ones(1, f32))
    self.decay = const(decay, f32)
    self.buffer = np.zeros(n*2+1, 'f4')
    self.size_splits = const([n, n, 1], i32)

    self.mean = var(zeros(n, f32))
    self.std = var(ones(n, f32))

    self.fixed = False

  @tf.function
  def inc(self, newval):
    newsum, newsumsq, newcount = split(newval, self.size_splits)
    decay = self.decay
    self._sum.assign(self._sum     * decay + newsum)
    self._sumsq.assign(self._sumsq * decay + newsumsq)
    self._cnt.assign(self._cnt     * decay + newcount)

    self.mean.assign(div(self._sum,self._cnt))
    self.std.assign(sqrt(maximum(div(self._sumsq,self._cnt) - sq(self.mean), 1e-8)))

  def update(self, x):
    if not self.fixed:
      total = self.buffer
      addvec = compute_vec(x)
      #MPI.COMM_WORLD.Allreduce(addvec, total, op=MPI.SUM)
      self.inc(addvec)

  def assign(self, x):
    mean = 𝔼(x,0)
    std = rstd(x,0)

    self.mean.assign(mean)
    self.std.assign(maximum(std, 1e-4))
    self.fixed = True

  @tf.function
  def nrm(self, x):
    return (x - self.mean) / self.std

  @property
  def vars(self):
    return [self.mean, self.std]

rms = None
r_rms = None

def init():
  global rms, r_rms
  rms = RMS(ns)
  r_rms = RMS(1)
