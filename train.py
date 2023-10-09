import time
import os.path as osp
from util import log
import numpy as np
import tensorflow as tf
from random import shuffle
from util.act import generate_τ, generate_τ2
from util.opt import Adam
from const import α0, αT, γ, q, k
from loss import act, fwd
from util.pd import logp, ent
from util.tf_util import *
from util.rms import rms, r_rms
import gc


np_f32 = np.float32
np_mean = np.mean

if q == 1.0:
  from util.pd import rs as r
else:
  from util.pd import rt as r

def train(alg, env, eval_env, π, buf, D, expert_dataset, batch_size, *,
     π_step, d_step,
     lr_π=3e-4, lr_d=3e-4,
     max_steps=0, expr_path='', record_intvl=10, save_intvl=-1, 
     burnin_steps=0):

  is_2d = "Multi" in env.name
  if is_2d:
    lr_π = 5e-4
    lr_d = 5e-4

  Opt = lambda x: Adam(x, β1=0.9, β2=0.999)
  DOpt = lambda x: Adam(x, β1=0.9, β2=0.999)

  from loss import πv_vars
  πopt = Opt(πv_vars)

  do_irl = D is not None

  τ_sampler = generate_τ(π, env, batch_size)
  eval_sampler = generate_τ(π, eval_env, 4000)

  get_τ = τ_sampler.__next__
  evaluate = eval_sampler.__next__

  get_batch = buf.get_next_batch
  if do_irl:
    get_batchπ = buf.get_next_batch_adv
    get_batchE = expert_dataset.get_next_batch
    from loss import d_vars
    dopt = DOpt(d_vars)
    rwd = D.rwd
    nintype = len(D.intype)
    @tf.function
    def tf_locals(πvals, dvals):
      return tf.concat([𝔼(πvals, 0), 𝔼(dvals, 0)], 0)
  else:
    @tf.function
    def tf_locals(πvals):
      return 𝔼(πvals, 0)

  from loss.πloss import πvgradvals
  if do_irl:
    from loss.dloss import dgradvals

  ep_total = 0
  steps = 0
  i = 0
  l = 0

  lr_π_vec = lr_π * ones(add_n([rprod(x)
      for x in shape_n(π.vars)]), f32)
  lr_q_vec = lr_d * ones(add_n([rprod(x)
      for x in shape_n(π.q1_vars)]), f32)

  ep_cnt = 0
  ep_lens = []
  ep_rets = []
  r_trues = []

  πvals = []
  dvals = []

  keys = ['ep/ep_cnt', 'ep/ep_len', 'ep/ep_ret', 'ep/r_true']
  if alg == 'mdirl':
    keys += ['pi/alpha']
  keys += ['pi/loss','pi/entropy',
      'value/q_loss', 'value/r_mean']
  if do_irl:
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

    if do_irl:
      for j in range(d_step):
        xπ = get_batchπ(batch_size)
        xE = get_batchE(batch_size)
        if alg == 'mdirl':
          dgrad, dval = dgradvals(*xπ, *xE, αt)
        else:
          dgrad, dval = dgradvals(*xπ, *xE)

        del(xπ)
        del(xE)

        dopt.update(dgrad, lr_d)

        del(dgrad)
        dvals.append(dval)

    for j in range(π_step):
      if steps % 100 == 0:
        τ = get_τ()
        buf.add(batch_size, τ)

      x = get_batch(batch_size)

      if do_irl:
        r = rwd(*x[:nintype])
        x = x[:4]+[r]


      πvgrad, πvval, reg_rewards = πvgradvals(*x)

      if not(train_π):
        r_rms.update(reg_rewards[...,np.newaxis])
      else:
        if steps % 10000 <= 5:
          r_rms.update(reg_rewards[...,np.newaxis])

      if train_π:
        πopt.update(πvgrad, lr_π)
        del(πvgrad)
      πvals.append(πvval)
      π.updatevt(x[-1])

      del(x)

      steps += 1

      if save_intvl > 0 and (steps - burnin_steps) >= (l * save_intvl):
        param_dict = {}
        for idx, var in enumerate(π.vars):
          param_dict[f'pi_{idx}'] = var.numpy()
        for idx, var in enumerate(r_rms.vars):
          param_dict[f'r_rms_{idx}'] = var.numpy()
        if D is not None:
          for idx, var in enumerate(D.vars):
            param_dict[f'd_{idx}'] = var.numpy()
        param_path = osp.join(expr_path, f'param_{l * save_intvl}.npz')
        with open(param_path, 'wb') as f:
          np.savez(f, **param_dict)
        l += 1

    if (i%(record_intvl//π_step)) == 0:
      evals = evaluate()
      ep_cnt += len(evals['ep_len'])
      ep_lens.extend(evals['ep_len'])
      ep_rets.extend(evals['ep_ret'])
      r_trues.append(np_mean(evals['r_true'], dtype=np_f32))
      ep_total += ep_cnt

      if do_irl:
        args = [πvals, dvals]
      else:
        args = [πvals]
      if alg == 'mdirl':
        vals = np.concatenate([[
            np_f32(ep_cnt),
            np_mean(ep_lens, dtype=np_f32),
            np_mean(ep_rets, dtype=np_f32),
            np_mean(r_trues, dtype=np_f32), αt],
            tf_locals(*args)])
      else:
        vals = np.concatenate([[
          np_f32(ep_cnt),
          np_mean(ep_lens, dtype=np_f32),
          np_mean(ep_rets, dtype=np_f32),
          np_mean(r_trues, dtype=np_f32)],
          tf_locals(*args)])

      for key, val in zip(keys, vals):
        log.record_tab(key, val)
      log.record_tab('ep/ep_total', ep_total)
      log.record_tab('ep/steps', steps)
      log.dump_tab(i//(record_intvl//π_step), time.time() - start_time)

      ep_cnt = 0
      ep_lens.clear()
      ep_rets.clear()
      r_trues.clear()

      πvals.clear()
      dvals.clear()

    gc.collect()
    i += 1

    if max_steps and (steps - burnin_steps) >= max_steps:
      break
