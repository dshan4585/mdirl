import tensorflow as tf
from const_discrete import reg_type, na, q
from util.tf_util_discrete import *
from util.pd_discrete import ent, tent,

from util.pd_discrete import rf as r
from util.pd_discrete import breg as D

class RAIRLDiscrete:
  def __init__(self, π):
    self.π = π
    rinit = tf.zeros(na,tf.float32)
    self.rlogit = tf.Variable(rinit, dtype=tf.float32)
    self.vars = [self.rlogi]

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

    f = logp(a,rlogit) - logp(a,logit)
    rπ, rE = split(r(a,rlogit),2)

    loss = 2*𝔼(bce(l,f))

    return loss, rπ, rE