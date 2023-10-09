import tensorflow as tf
import numpy as np
from const import a_limit, a_scale, u_scale
from util.tf_util import *
from util.rms import rms

np_concat = np.concatenate
np_randint = np.random.randint
np_i32 = np.int32

class DataSet:
  def get_next_batch(self, size):
    return self._get_next_batch(np_randint(0, self.n, size, dtype=np_i32))
  @tf.function
  def _get_next_batch(self, indices):
    return split(gather(self.data, indices), self.size_splits, -1)

class SA(DataSet):
  def __init__(self, s, a):
    self.n = len(s)
    self.data = const(np_concat((s,a),-1),f32)
    self.size_splits = const([s.shape[-1],a.shape[-1]],i32)

class SAS(DataSet):
  def __init__(self, s, a, ś):
    self.n = len(s)
    self.data = const(np_concat((s,a,ś),-1),f32)
    self.size_splits = const([s.shape[-1],a.shape[-1],s.shape[-1]],i32)

def get_trj(env_id, path, intype):
  τ_data = np.load(path)
  s = τ_data['s'].copy()

  a_f64 = np.float64(τ_data['a'].copy())
  us_f64 = np.float64(u_scale)
  al_f64 = np.float64(a_limit)
  as_f64 = np.float64(a_scale)
  y = np.clip(a_f64/as_f64,-1.,1.)/al_f64
  a = np.float32(np.arctanh(y)/us_f64)
  ś = τ_data['ś'].copy()

  if env_id == 'Ant-v3':
    s = s[:,:27]
    ś = ś[:,:27]

  if intype == 'sa':
    d = SA(s, a)
  elif intype == 'saś':
    d = SAS(s, a, ś)
  else:
    raise NotImplementedError
  G = τ_data['ep_rets']
  avg_ret = sum(G)/len(G)
  std_ret = np.std(np.array(G))
  τ_data.close()

  return d
