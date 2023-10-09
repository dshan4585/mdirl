import tensorflow as tf
from nn.layer import Dense, fwda, fwda1, fwdb, fwdb1, fwdc, soft_update
from const import ns, na, nh, a_limit, u_limit, a_scale
from util.rms import rms
from util.pd import logp
from util.tf_util import *

class DiagGaussianPolicyValue:
  def __init__(self):
    self.π_net = π_net = []
    self.q1_net = q1_net = []
    self.q2_net = q2_net = []
    self.q1t_net = q1t_net = []
    self.q2t_net = q2t_net = []

    d = na

    π_net += [Dense(ns,nh)]
    π_net += [Dense(nh,nh)]
    π_net += [Dense(nh,2*d)]

    sz = list(reversed(range(d)))
    self.sz = const(sz,i32)
    self.duts1=[const([0]*(d-1-i)+[1],f32)for i in sz]
    self.duts=[const([[0]*(d-1-i)+[1]],f32)for i in sz]

    q1_net += [Dense(ns+na,nh)]
    q1_net += [Dense(nh,nh)]
    q1_net += [Dense(nh,1)]

    q2_net += [Dense(ns+na,nh)]
    q2_net += [Dense(nh,nh)]
    q2_net += [Dense(nh,1)]

    q1t_net += [Dense(ns+na,nh)]
    q1t_net += [Dense(nh,nh)]
    q1t_net += [Dense(nh,1)]

    q2t_net += [Dense(ns+na,nh)]
    q2t_net += [Dense(nh,nh)]
    q2t_net += [Dense(nh,1)]

    self.vars = self._vars
    self.q1_vars = self._q1_vars
    self.q2_vars = self._q2_vars
    self.q1t_vars = self._q1t_vars
    self.q2t_vars = self._q2t_vars

    for v_var, vt_var in zip(self.q1_vars+self.q2_vars,self.q1t_vars+self.q2t_vars):
      vt_var.assign(v_var)

    self.coeff = const(0.005, f32)

  @tf.function
  def fwd(self, s):
    n = shape(s)[0]
    o1, o2 = split(fwda(s, *self.vars),2,-1)
    μ = o1
    lnσ = log_tanh(o2)

    return μ, lnσ

  @tf.function
  def fwd1(self, s):
    o1, o2 = split(fwda1(s, *self.vars),2,-1)
    μ = o1
    lnσ = log_tanh(o2)

    return μ, lnσ

  @tf.function
  def fwdq1(self, s, a):
    return reshape(fwdc(tf.concat([s,a],-1), *self.q1_vars),[-1])

  @tf.function
  def fwdq2(self, s, a):
    return reshape(fwdc(tf.concat([s,a],-1), *self.q2_vars),[-1])

  @tf.function
  def fwdq1t(self, s, a):
    return reshape(fwdc(tf.concat([s,a],-1), *self.q1t_vars),[-1])

  @tf.function
  def fwdq2t(self, s, a):
    return reshape(fwdc(tf.concat([s,a],-1), *self.q2t_vars),[-1])

  @tf.function
  def updatevt(self, r):
    coeff = self.coeff
    soft_update(coeff, *self.q1_vars, *self.q1t_vars)
    soft_update(coeff, *self.q2_vars, *self.q2t_vars)

  @tf.function
  def act(self, s):
    μ, lnσ = self.fwd1(s)
    a = μ+exp(lnσ)*normal(shape(μ))
    return a

  @tf.function
  def act_batch(self, s):
    μ, lnσ = self.fwd(s)
    return μ+exp(lnσ)*normal(shape(μ))

  @property
  def _vars(self):
    ret = []
    for layer in self.π_net:
      ret.extend(layer.vars)
    return ret

  @property
  def _q1_vars(self):
    ret = []
    for layer in self.q1_net:
      ret.extend(layer.vars)
    return ret

  @property
  def _q2_vars(self):
    ret = []
    for layer in self.q2_net:
      ret.extend(layer.vars)
    return ret

  @property
  def _q1t_vars(self):
    ret = []
    for layer in self.q1t_net:
      ret.extend(layer.vars)
    return ret

  @property
  def _q2t_vars(self):
    ret = []
    for layer in self.q2t_net:
      ret.extend(layer.vars)
    return ret