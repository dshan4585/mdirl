import tensorflow as tf
from nn.layer import Dense, fwda, fwda1, fwdb, fwdb1, fwdc, soft_update
from const import ns, na, nh, a_limit, a_scale
from util.rms import rms
from util.tf_util import *

class DiagGaussianPolicy:
  def __init__(self):
    self.μ_net = μ_net = []
    self.lnσ_net = lnσ_net = []

    μ_net += [Dense(ns,nh)]
    μ_net += [Dense(nh,nh)]
    μ_net += [Dense(nh,na)]

    lnσ_net += [Dense(ns,nh)]
    lnσ_net += [Dense(nh,nh)]
    lnσ_net += [Dense(nh,na)]

    self.vars = self._vars

    net_vars = self._net_vars
    self.μ_vars = net_vars[0]
    self.lnσ_vars = net_vars[1]

    self.coeff = const(0.005, f32)

  @tf.function
  def fwd(self, s):
    n = shape(s)[0]
    μ = fwda(s, *self.μ_vars)
    lnσ = log_tanh(fwda(s, *self.lnσ_vars))
    return μ, lnσ

  @tf.function
  def fwd1(self, s):
    μ = fwda1(s, *self.μ_vars)
    lnσ = log_tanh(fwda1(s, *self.lnσ_vars))
    return μ, lnσ

  @tf.function
  def act(self, s):
    μ, lnσ = self.fwd1(s)
    return μ+exp(lnσ)*normal(shape(μ))

  @tf.function
  def act_batch(self, s):
    μ, lnσ = self.fwd(s)
    return μ+exp(lnσ)*normal(shape(μ))

  @property
  def _vars(self):
    ret = []
    for layer in self.μ_net + self.lnσ_net:
      ret.extend(layer.vars)
    return ret

  @property
  def _net_vars(self):
    ret = []
    for net in [self.μ_net, self.lnσ_net]:
      net_ret = []
      for layer in net:
        net_ret.extend(layer.vars)
      ret.append(net_ret)
    return ret

  @tf.function
  def updateπ(self, π):
    coeff = self.coeff
    soft_update(coeff, *π.μ_vars, *self.μ_vars)
    soft_update(coeff, *π.lnσ_vars, *self.lnσ_vars)
