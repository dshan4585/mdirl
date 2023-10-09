import tensorflow as tf
from const_discrete import reg_type, na, q
from util.tf_util_discrete import *
from util.pd_discrete import ent, tent,

from util.pd_discrete import rf as r
from util.pd_discrete import breg as D

class MirrorDescentDiscrete:
  def __init__(self, π):
    self.π = π
    rinit = tf.zeros(na,tf.float32)
    tinit = tf.zeros(na,tf.float32)
    self.rlogit = tf.Variable(rinit, dtype=tf.float32)
    self.tlogit = tf.Variable(tinit, dtype=tf.float32)
    self.vars = [self.rlogit, self.tlogit]
    self.updater()

  @tf.function
  def rwd(self, a):
    return r(a,self.rlogit)

  @tf.function
  def loss(self, aπ, aE, α):
    nπ = shape(sπ)[0]
    nE = shape(sE)[0]
    lπ = zeros(nπ)
    lE = ones(nE)
    l = concat([lπ,lE],0)
    a = concat([aπ,aE],0)

    logit = self.π.logit
    rlogit = self.rlogit
    tlogit = self.tlogit

    f = logp(a,tlogit) - logp(a,logit)
    rπ, rE = split(r(a,rlogit),2)

    loss = 2*𝔼(bce(l,f))

    rloss = 𝔼((1/α)*D(rlogit,no_grad(tdist)) +\
        ((α-1)/α)*D(rlogit,dist))

    return loss+rloss, rπ, rE

  def updater(self):
    for π_var, r_var in zip(self.π.vars, self.r_vars):
      r_var.assign(π_var)