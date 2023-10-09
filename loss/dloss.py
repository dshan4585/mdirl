import tensorflow as tf
from loss import d_vars, d_fn
from util.tf_util import *

@tf.function
def dgradvals(*args):
  loss, rπ, rE = d_fn(*args)
  return concat([reshape(g,[-1]) for g in grad(loss,d_vars)],-1), stack([loss, 𝔼(rπ), 𝔼(rE)])

@tf.function
def dgradvals2(*args):
  loss, rπ, rE, div = d_fn(*args)
  return concat([reshape(g,[-1]) for g in grad(loss,d_vars)],-1), stack([loss, 𝔼(rπ), 𝔼(rE), 𝔼(div)])