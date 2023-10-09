import tensorflow as tf
from nn.layer import Dense, BatchNorm, fwdf
from const import ns, na, nh
from util.rms import rms
from util.tf_util import *
from util.pd import sample
from loss import fwd

class DAC:
  intype = 'sa'
  def __init__(self):
    self.net = net = []
    net += [Dense(ns+na,nh)]
    net += [Dense(nh,nh)]
    net += [Dense(nh,1)]
    self.vars = self._vars

  @tf.function
  def fwd(self, s, a):
    return reshape(fwdf(concat([s,activ(a)],-1),*self.vars),[-1])

  @tf.function
  def rwd(self, *args):
    return self.fwd(*args)

  @tf.function
  def loss(self, sπ, aπ, sE, aE):
    nπ = shape(sπ)[0]
    nE = shape(sE)[0]
    lπ = zeros(nπ)
    lE = ones(nE)
    l = concat([lπ,lE],0)
    s = concat([sπ,sE],0)
    a = concat([aπ,aE],0)
    f = self.fwd(s,a)
    loss = 2*𝔼(bce(l,f))
    fπ,fE = split(f,2)

    u = uniform((nπ,1))
    si = u * sπ + (1-u) * sE
    ai = u * aπ + (1-u) * aE
    fi = self.fwd(si,ai)
    gi = concat(grad(fi,[si,ai]),-1)
    gp = 𝔼(rsum(sq(gi),-1))

    return loss+1e-3*gp, softplus(fπ), softplus(fE)

  @property
  def _vars(self):
    ret = []
    for layer in self.net:
      ret.extend(layer.vars)
    return ret