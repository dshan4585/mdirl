import time
import os.path as osp
from util import log
import numpy as np
import tensorflow as tf
from util.opt import Adam
from const_discrete import α0, αT, na
from util.tf_util_discrete import *
from util.pd_discrete import breg, rf, pot, logp

np_f32 = np.float32
np_mean = np.mean

def train(alg, π, πE, D, batch_size, *,
     π_step, d_step,
     lr_π=1e-2, lr_d=1e-2,
     max_steps=0, expr_path='', record_intvl=10, save_intvl=-1, 
     burnin_steps=0):
  Opt = lambda x: Adam(x, β1=0.5, β2=0.99)
  DOpt = lambda x: Adam(x, β1=0.9, β2=0.99)
  opt = Opt(π.vars)
  dopt = DOpt(D.vars)
  rwd = D.rwd

  @tf.function
  def tf_locals(dvals):
    return tf.concat([𝔼(dvals, 0)], 0)

  logit = π.logit
  @tf.function
  def gradvals(a):
    return grad(na *(pot(logit) - 𝔼(logp(a,logit)*rf(a,D.rlogit))),[logit])[0]

  @tf.function
  def dgradvals(*args):
    loss, rπ, rE = D.loss(*args)
    return concat([reshape(g,[-1]) for g in grad(na*loss,D.vars)],-1), stack([loss, 𝔼(rπ), 𝔼(rE)])

  steps = 0
  i = 0
  l = 0

  dvals = []
  keys = []
  if alg == 'mdirl':
    keys += ['pi/div', 'pi/rev','pi/rdiv', 'pi/rrev','pi/tdiv', 'pi/trev', 'pi/alpha']
  else:
    keys += ['pi/div', 'pi/rev','pi/rdiv', 'pi/rrev']
  keys += ['disc/loss', 'disc/r', 'disc/rE']

  if alg == 'mdirl':
    assert αT >= α0

  αt = const(α0, f32)
  train_π = False

  while True:
    if not(train_π) and steps >= burnin_steps:
      train_π = True

    start_time = time.time()

    t_rate = np.maximum(steps - burnin_steps, 0) / max_steps
    αt = tf.constant(t_rate * (αT - α0) + α0, f32)

    for j in range(d_step):
      xπ = π.act_batch(batch_size)
      xE = πE.act_batch(batch_size)
      if alg == 'mdirl':
        dgrad, dval = dgradvals(*[xπ, xE, αt])
      else:
        dgrad, dval = dgradvals(*[xπ, xE])
      dopt.update(dgrad, lr_d)
      dvals.append(dval)

    π.logit.assign(D.rlogit)
    for j in range(π_step):
      xπ = π.act_batch(batch_size)
      opt.update(gradvals(xπ), lr_π)

    if save_intvl > 0 and (steps - burnin_steps) >= (l * save_intvl):
      param_dict = {}
      for idx, var in enumerate(π.vars):
        param_dict[f'pi_{idx}'] = var.numpy()
      for idx, var in enumerate(πE.vars):
        param_dict[f'piE_{idx}'] = var.numpy()
      if D is not None:
        for idx, var in enumerate(D.vars):
          param_dict[f'd_{idx}'] = var.numpy()
      param_path = osp.join(expr_path, f'param_{l * save_intvl}.npz')
      with open(param_path, 'wb') as f:
        np.savez(f, **param_dict)
      l += 1

    if (i%record_intvl) == 0:

      args = [dvals]
      if alg == 'mdirl':
        vals = np.concatenate([[
            breg(π.logit, πE.logit),breg(πE.logit, π.logit),
            breg(D.rlogit, πE.logit),breg(πE.logit, D.rlogit),
            breg(D.tlogit, πE.logit),breg(πE.logit, D.tlogit),αt],
            tf_locals(dvals)])
      else:
        vals = np.concatenate([[
            breg(π.logit, πE.logit),breg(πE.logit, π.logit),
            breg(D.rlogit, πE.logit),breg(πE.logit, D.rlogit),],tf_locals(dvals)])

      for key, val in zip(keys, vals):
        log.record_tab(key, val)
      log.record_tab('ep/steps', steps)
      log.dump_tab(i//record_intvl, time.time() - start_time)

      dvals.clear()

    i += 1
    steps += π_step

    if max_steps and (steps - burnin_steps) >= max_steps:
      break