import numpy as np
import tensorflow as tf
from const_discrete import na
from util.tf_util_discrete import *

class DiscreteStateless:
  def __init__(self):
    init = tf.zeros(na, tf.float32)
    self.logit = tf.Variable(init, dtype=tf.float32)
    self.vars = [self.logit]

  @tf.function
  def act(self):
    a = categorical(ed(self.logit,0),1,tf.int32)
    return squeeze(a)

  @tf.function
  def act_batch(self, n):
    return reshape(categorical(ed(self.logit,0),n,tf.int32),[-1])
from util.pd_discrete import logp
from util.pd_discrete import rf as r
from util.pd_discrete import breg as D

class RAIRLDiscrete:
  def __init__(self, π):
    self.π = π
    rinit = tf.zeros(na,tf.float32)
    self.rlogit = tf.Variable(rinit, dtype=tf.float32)
    self.vars = [self.rlogit]

  @tf.function
  def rwd(self, a):
    return r(a,self.rlogit)

  @tf.function
  def loss(self, aπ, aE):
    nπ = shape(aπ)[0]
    nE = shape(aE)[0]
    lπ = zeros(nπ)
    lE = ones(nE)
    l = concat([lπ,lE],0)
    a = concat([aπ,aE],0)

    logit = self.π.logit
    rlogit = self.rlogit

    f = r(a,rlogit) - r(a,logit)
    rπ, rE = split(r(a,rlogit),2)

    loss = 2*𝔼(bce(l,f))

    return loss, rπ, rE

class MirrorDescentDiscrete:
  def __init__(self, π):
    self.π = π
    rinit = tf.zeros(na, tf.float32)
    tinit = tf.zeros(na, tf.float32)
    self.rlogit = tf.Variable(rinit, dtype=tf.float32)
    self.tlogit = tf.Variable(tinit, dtype=tf.float32)
    self.vars = [self.rlogit, self.tlogit]

  @tf.function
  def rwd(self, a):
    return r(a,self.rlogit)

  @tf.function
  def loss(self, aπ, aE, α):
    nπ = shape(aπ)[0]
    nE = shape(aE)[0]
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

    rloss = 𝔼((1/α)*D(rlogit,no_grad(tlogit)) +\
        ((α-1)/α)*D(rlogit,logit))

    return loss+rloss, rπ, rE
