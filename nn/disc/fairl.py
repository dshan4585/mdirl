import tensorflow as tf
from nn.layer import Dense, fwde
from const import ns, na, nh
from util.rms import rms
from util.tf_util import *

class FAIRL:
  intype = 'sa'
  def __init__(self):
    self.net = net = []
    net += [Dense(ns+na,nh)]
    net += [Dense(nh,nh)]
    net += [Dense(nh,1)]
    self.vars = self._vars

  @tf.function
  def fwd(self, s, a):
    return reshape(fwde(concat([rms.nrm(s),a],-1), *self.vars),[-1])

  @tf.function
  def rwd(self, *args):
    f = self.fwd(*args)
    c = clip(f, -5., 5.)
    return exp(c)

  @tf.function
  def loss(self, sπ, aπ, sE, aE):
    nπ = shape(sπ)[0]
    nE = shape(sE)[0]
    lπ = zeros(nπ)
    lE = ones(nE)
    l = concat([lπ,lE],0)
    fπ = self.fwd(sπ,aπ)
    fE = self.fwd(sE,aE)
    loss = 2*𝔼(bce(l,tf.concat([fπ,fE],0)))
    cπ = clip(fπ, -10., 10.)
    cE = clip(fE, -10., 10.)
    return loss, - cπ*exp(cπ), - cE*exp(cE)

  @property
  def _vars(self):
    ret = []
    for layer in self.net:
      ret.extend(layer.vars)
    return ret