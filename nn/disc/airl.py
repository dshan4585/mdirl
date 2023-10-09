import tensorflow as tf
from nn.layer import Dense, fwde, fwdf
from const import γ, ns, na, nh, k
from util.rms import rms
from util.tf_util import *
from loss import act, fwd, π_vars
from util.pd import logp, log_activ, ent, sample

class AIRL:
  intype = 'saś'
  def __init__(self, π):
    self.π = π
    self.g_net = g_net = []
    self.h_net = h_net = []

    ng_in = ns+na

    g_net += [Dense(ng_in,nh)]
    g_net += [Dense(nh,nh)]
    g_net += [Dense(nh,1)]

    h_net += [Dense(ns,nh)]
    h_net += [Dense(nh,nh)]
    h_net += [Dense(nh,1)]

    self.vars = self._vars
    self.g_vars = self._g_vars
    self.h_vars = self._h_vars

  @tf.function
  def fwdg(self, s, a):
    return reshape(fwdf(concat([s,activ(a)],-1), *self.g_vars),[-1])

  @tf.function
  def fwdh(self, s):
    return reshape(fwde(s, *self.h_vars),[-1])

  @tf.function
  def rwd(self, *args):
    s, a, ś = args
    dist = fwd(s)
    logπ = logp(a,*dist)+log_activ(a)

    h = self.fwdh(s)
    g = self.fwdg(s, a)
    h́ = self.fwdh(ś)

    f = g + γ * h́ - h
    return k * f

  @tf.function
  def loss(self, sπ, aπ, śπ, sE, aE, śE):
    aπ = no_grad(sample(*fwd(sπ)))

    nπ = shape(sπ)[0]
    nE = shape(sE)[0]
    lπ = zeros(nπ)
    lE = ones(nE)
    l = concat([lπ,lE],0)
    s = concat([sπ,sE],0)
    a = concat([aπ,aE],0)
    g = self.fwdg(s,a)
    h = self.fwdh(s)
    h́ = self.fwdh(s)

    dist = fwd(s)
    logπ = logp(a,*dist)+log_activ(a)
    f = g + γ * h́ - h - logπ
    fπ,fE = split(f,2)

    loss = 2*𝔼(bce(l,f))

    u = uniform((nπ,1))
    si = u * sπ + (1-u) * sE
    ai = u * aπ + (1-u) * aE
    fi = self.fwdg(si,ai)
    f2i = self.fwdh(si)
    gi = concat(grad(fi,[si,ai]),-1)
    g2i = grad(f2i,[si])[0]
    gp = 𝔼(rsum(sq(gi),-1))+𝔼(rsum(sq(g2i),-1))

    return loss+5e-2*gp, fπ, fE

  @property
  def _vars(self):
    ret = []
    for layer in self.g_net + self.h_net:
      ret.extend(layer.vars)
    return ret

  @property
  def _g_vars(self):
    ret = []
    for layer in self.g_net:
      ret.extend(layer.vars)
    return ret

  @property
  def _h_vars(self):
    ret = []
    for layer in self.h_net:
      ret.extend(layer.vars)
    return ret