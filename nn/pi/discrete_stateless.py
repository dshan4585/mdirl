import numpy as np
import tensorflow as tf
from const_discrete import na
from tf_util import *

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