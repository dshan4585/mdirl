import tensorflow as tf
from nn.layer import Dense, ConvQuad, CELU, PosDense, fwdd, fwde, fwdf, conv, soft_update
from const import ns, na, nh, u_limit, a_limit, a_scale, q, k, k2, gp_coeff
from util.tf_util import *
from util.rms import rms
from util.pd import ent, tent, ent2, tent2, logp, log_activ, sample
from loss import act, fwd, π_vars

if q == 1.0:
  from util.pd import rs as r
  from util.pd import kl as D
else:
  from util.pd import rt as r
  from util.pd import tkl as D

k = const(k, f32)

class MirrorDescentDiagGaussian2:
  intype = 'sa'
  def __init__(self, π):
    self.π = π
    self.μ_net = μ_net = []
    self.lnσ_net = lnσ_net = []
    self.μt_net = μt_net = []
    self.lnσt_net = lnσt_net = []
    self.b_net = b_net = []

    d = na

    μ_net += [Dense(ns,nh)]
    μ_net += [Dense(nh,nh)]
    μ_net += [Dense(nh,d)]

    lnσ_net += [Dense(ns,nh)]
    lnσ_net += [Dense(nh,nh)]
    lnσ_net += [Dense(nh,na)]

    μt_net += [Dense(ns,nh)]
    μt_net += [Dense(nh,nh)]
    μt_net += [Dense(nh,d)]

    lnσt_net += [Dense(ns,nh)]
    lnσt_net += [Dense(nh,nh)]
    lnσt_net += [Dense(nh,na)]

    b_net += [Dense(ns,nh)]
    b_net += [Dense(nh,nh)]
    b_net += [Dense(nh,1)]

    self.vars = self._vars
    net_vars = self._net_vars
    self.μ_vars    = net_vars[0]
    self.lnσ_vars  = net_vars[1]

    self.μt_vars   = net_vars[2]
    self.lnσt_vars = net_vars[3]
    self.b_vars =    net_vars[4]

    self.updater()

  @tf.function
  def fwdr(self, s):
    n = shape(s)[0]

    μ = tanh(fwdd(s, *self.μ_vars)/u_limit)*u_limit
    lnσ = log_tanh(fwdd(s, *self.lnσ_vars))
    return μ, lnσ

  @tf.function
  def fwdt(self, s):
    n = shape(s)[0]

    μ = tanh(fwdd(s, *self.μt_vars)/u_limit)*u_limit
    lnσ = log_tanh(fwdd(s, *self.lnσt_vars))

    return μ, lnσ

  @tf.function
  def fwdb(self, s):
    return reshape(fwdf(s,*self.b_vars),[-1])

  @tf.function
  def rwd(self, s, a):
    return k*r(a, *self.fwdr(s)) + k2*self.fwdb(s)

  @tf.function
  def loss(self, sπ, aπ, sE, aE, α):
    aπ = no_grad(sample(*fwd(sπ)))

    nπ = shape(sπ)[0]
    nE = shape(sE)[0]
    lπ = zeros(nπ)
    lE = ones(nE)
    l = concat([lπ,lE],0)
    s = concat([sπ,sE],0)
    a = concat([aπ,aE],0)

    dist = fwd(s)
    rdist = self.fwdr(s)
    tdist = self.fwdt(s)
    b = self.fwdb(s)

    f = logp(a,*tdist) - logp(a,*dist) + no_grad(b)
    rπ, rE = split(r(a, *rdist) + b,2)

    loss = 2*𝔼(bce(l,f))+2*𝔼(bce(l,b))

    rloss = 𝔼((1/α)*D(*rdist,*map_structure(no_grad,tdist)) +\
        ((α-1)/α)*D(*rdist,*dist))

    u = uniform((nπ,1))
    si = u * sπ + (1-u) * sE
    ai = u * aπ + (1-u) * aE

    fi = logp(ai, *self.fwdr(si))
    f2i = logp(ai, *self.fwdt(si))
    f3i = self.fwdb(si)

    gi = grad(fi,[si])[0]
    g2i = grad(f2i,[si])[0]
    g3i = grad(f3i,[si])[0]
    gp = 0.5*(𝔼(rsum(sq(gi),-1))+𝔼(rsum(sq(g2i),-1))+𝔼(rsum(sq(g3i),-1)))

    return loss+rloss+gp_coeff*gp, rπ, rE

  @property
  def _vars(self):
    ret = []
    for layer in self.μ_net + self.lnσ_net + self.μt_net + self.lnσt_net + self.b_net:
      ret.extend(layer.vars)
    return ret

  @property
  def _net_vars(self):
    ret = []
    for net in [self.μ_net, self.lnσ_net, self.μt_net, self.lnσt_net, self.b_net]:
      net_ret = []
      for layer in net:
        net_ret.extend(layer.vars)
      ret.append(net_ret)
    return ret

  def updater(self):
    for π_var, r_var in zip(self.π.μ_vars, self.μ_vars):
      r_var.assign(π_var)
    for π_var, r_var in zip(self.π.lnσ_vars, self.lnσ_vars):
      r_var.assign(π_var)