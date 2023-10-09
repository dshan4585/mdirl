import random
import numpy as np
import tensorflow as tf

def set_seed(i, env, envE):
  rank = 0
  myseed = i  + 10000 * rank if i is not None else None
  random.seed(myseed)
  np.random.seed(myseed)
  tf.random.set_seed(myseed)
  env.seed(myseed)
  envE.seed(myseed)
