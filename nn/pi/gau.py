import tensorflow as tf
from nn.layer import Dense, fwda, fwda1, fwdb, fwdb1, fwdc, soft_update
from const import ns, na, nh, a_limit, u_limit, a_scale, u_scale
from util.rms import rms
from util.pd import logp
from util.tf_util import *

class GaussianPolicy:
  def __init__(self):
    self.π_net = π_net = []

    d = na
    self.ult_dim = d*(d-1)//2

    π_net += [Dense(ns,nh)]
    π_net += [Dense(nh,nh)]
    π_net += [Dense(nh,d*(d+3)//2)]

    sz = list(reversed(range(d)))
    self.sz = const(sz,i32)
    self.duts=[const([[0]*(d-1-i)+[1]],f32)for i in sz]

    self.vars = self._vars

    self.coeff = const(0.005, f32)

  @tf.function
  def fwd(self, s):
    n = shape(s)[0]
    o1, o2, o3 = split(fwda(s, *self.vars),[na,na,self.ult_dim],-1)
    μ = tanh(o1/u_limit)*u_limit
    lnσ = log_tanh(o2)
    ult = stack([concat([tile(d,[n,1]),l],-1)
        for d, l in zip(self.duts,
        split(u_scale*o3,self.sz,-1))],-1)

    return μ, lnσ, ult


  @property
  def _vars(self):
    ret = []
    for layer in self.π_net:
      ret.extend(layer.vars)
    return ret