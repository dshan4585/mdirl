import tensorflow as tf
from nn.layer import Dense, fwdd, fwde, fwdf, fwdg
from const import ns, na, nh, a_limit, u_limit, a_scale, q, k, k2, gp_coeff
from util.rms import rms
from util.tf_util import *
from loss import act, fwd, fwd, π_vars
from util.pd import sample, log_activ, kl_std, logp

if q == 1.0:
  from util.pd import rs as r
  from util.pd import dfs as df
  from util.pd import kl as D
else:
  from util.pd import rt as r
  from util.pd import dft as df
  from util.pd import tkl as D

class RAIRLDiagGaussian:
  intype = 'sa'
  def __init__(self, π):
    self.π = π
    self.r_net = r_net = []
    self.b_net = b_net = []

    d = na

    r_net += [Dense(ns,nh)]
    r_net += [Dense(nh,nh)]
    r_net += [Dense(nh,d*2)]

    b_net += [Dense(ns,nh)]
    b_net += [Dense(nh,nh)]
    b_net += [Dense(nh,1)]

    self.vars = self._vars
    self.net_vars = self._net_vars

    self.r_vars = self.net_vars[0]
    self.b_vars = self.net_vars[1]

  @tf.function
  def fwdr(self, s):
    n = shape(s)[0]
    o1, o2 = split(fwdd(s, *self.r_vars),2,-1)

    μ = tanh(o1/u_limit)*u_limit
    lnσ = log_tanh(o2)
    return μ, lnσ

  @tf.function
  def fwdb(self, s):
    return reshape(fwdf(s,*self.b_vars),[-1])

  @tf.function
  def rwd(self, s, a):
    return k*df(a, *self.fwdr(s)) + k2*self.fwdb(s)

  @tf.function
  def loss(self, sπ, aπ, sE, aE):
    aπ = no_grad(sample(*fwd(sπ)))

    nπ = shape(sπ)[0]
    nE = shape(sE)[0]
    lπ = zeros(nπ)
    lE = ones(nE)
    l = concat([lπ,lE],0)
    s = concat([sπ,sE],0)
    a = concat([aπ,aE],0)

    dist = fwd(s)
    tdist = self.fwdr(s)
    b = self.fwdb(s)
    t = r(a, *dist)
    rwd = (df(a, *self.fwdr(s)) + no_grad(b))
    f = rwd - t
    rπ, rE = split(rwd,2)

    loss = 2*𝔼(bce(l,f))+2*𝔼(bce(l,b))

    u = uniform((nπ,1))
    si = u * sπ + (1-u) * sE
    ai = u * aπ + (1-u) * aE

    fi = logp(ai, *self.fwdr(si))
    f2i = self.fwdb(si)
    gi = grad(fi,[si])[0]
    g2i = grad(f2i,[si])[0]
    gp = .5*(𝔼(rsum(sq(gi),-1))+𝔼(rsum(sq(g2i),-1)))

    return loss+gp_coeff*gp, rπ, rE

  @property
  def _vars(self):
    ret = []
    for layer in self.r_net + self.b_net:
      ret.extend(layer.vars)
    return ret

  @property
  def _net_vars(self):
    ret = []
    for net in [self.r_net, self.b_net]:
      net_ret = []
      for layer in net:
        net_ret.extend(layer.vars)
      ret.append(net_ret)
    return ret


