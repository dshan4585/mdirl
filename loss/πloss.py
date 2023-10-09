import tensorflow as tf
from util.pd import ent, tent, ent2, tent2, logp, log_activ, sample, sample_uniform, logp_uniform, p_uniform, reg, kl, kl_std
import time
from const import γ
from loss import π_vars, v_vars, πv_vars, fwd, fwdq1, fwdq2, fwdq1t, fwdq2t, rwd
from const import ns, na, reg_lvl, a_limit, a_scale, k, q
from loss import act
from util.rms import rms, r_rms
from util.tf_util import *

γ_inv = const(1/(1-γ), f32)
γ_ = const(1-γ, f32)
γγ_ = const(γ-γ*γ, f32)
γ = const(γ, f32)
k = const(k, f32)
q_1 = const(q-1,f32)

if q == 1:
  if reg_lvl != 0:
    @tf.function
    def πvgradvals(s, a, ś, b, r):
      dist = fwd(s)
      e = ent(*dist)
      ke = k*e
      x = sample(*dist)
      logπx = logp(x,*dist)
      lnax = log_activ(x)
      logπx_activ = logπx+lnax
      klogπx = k*logπx
      klogπx_activ = k*logπx_activ
      πx = exp(logπx)
      πx_activ = exp(logπx_activ)

      qf1x = fwdq1(s,activ(x))
      qf2x = fwdq2(s,activ(x))

      qfx = minimum(qf1x,qf2x)
      π_loss = -𝔼(qfx-klogπx_activ)

      next_dist = fwd(ś)
      next_x = no_grad(sample(*next_dist))
      next_logπx = logp(next_x,*next_dist)
      next_lnax = log_activ(next_x)
      next_logπx_activ = next_logπx+next_lnax
      next_klogπx_activ = k*next_logπx_activ
      next_πx = exp(next_logπx)
      next_πx_activ = exp(next_logπx_activ)
      next_e = ent(*next_dist)
      next_ke = k*next_e

      next_qf1 = fwdq1t(ś,activ(next_x))
      next_qf2 = fwdq2t(ś,activ(next_x))
      next_qf = minimum(next_qf1,next_qf2)

      q_target = no_grad(r - r_rms.mean + where(b, 0., γ * (next_qf - next_klogπx_activ)))
      q1_loss = .5 * 𝔼(sq(fwdq1(s,activ(a)) - q_target))
      q2_loss = .5 * 𝔼(sq(fwdq2(s,activ(a)) - q_target))

      lnp_activ = logp(a,*dist)+log_activ(a)
      reg_rwd = r - k*lnp_activ

      v_loss = add_n([q1_loss, q2_loss])
      πvgrad = concat([reshape(g,[-1]) for g in\
          grad(π_loss, π_vars)+grad(v_loss, v_vars)],-1)
      vals = stack([π_loss, 𝔼(e), q1_loss, r_rms.mean[0]])

      return πvgrad, vals, reg_rwd
  else:
    @tf.function
    def πvgradvals(s, a, ś, b, r):
      dist = fwd(s)
      e = ent(*dist)
      ke = k*e
      x = sample(*dist)
      logπx = logp(x,*dist)
      klogπx = k*logπx
      πx = exp(logπx)

      qf1x = fwdq1(s,activ(x))
      qf2x = fwdq2(s,activ(x))

      qfx = minimum(qf1x,qf2x)

      π_loss = -𝔼(rwd(s,x)+qfx+ke)

      next_dist = fwd(ś)
      next_x = no_grad(sample(*next_dist))
      next_logπx = logp(next_x,*next_dist)
      next_klogπx = k*next_logπx

      next_qf1 = fwdq1t(ś,activ(next_x))
      next_qf2 = fwdq2t(ś,activ(next_x))
      next_qf = minimum(next_qf1,next_qf2)

      q_target = no_grad(where(b, 0., γ * (rwd(ś,next_x)-r_rms.mean+next_qf - next_klogπx)))
      q1_loss = .5 * 𝔼(sq(fwdq1(s,activ(a)) - q_target))
      q2_loss = .5 * 𝔼(sq(fwdq2(s,activ(a)) - q_target))

      reg_rwd = r +ke

      v_loss = add_n([q1_loss, q2_loss])
      πvgrad = concat([reshape(g,[-1]) for g in\
          grad(π_loss, π_vars)+grad(v_loss, v_vars)],-1)
      vals = stack([π_loss, 𝔼(e), q1_loss, r_rms.mean[0]])

      return πvgrad, vals, reg_rwd
else:
  if reg_lvl != 0:
    @tf.function
    def πvgradvals(s, a, ś, b, r):
      dist = fwd(s)
      e = ent(*dist)
      ke = k*e
      x = sample(*dist)
      logπx = logp(x,*dist)
      lnax = log_activ(x)
      logπx_activ = logπx+lnax
      klogπx = k*logπx
      kqlogπx_activ = k*(exp(q_1*logπx_activ)-1)

      πx = exp(logπx)
      πx_activ = exp(logπx_activ)

      qf1x = fwdq1(s,activ(x))
      qf2x = fwdq2(s,activ(x))

      qfx = minimum(qf1x,qf2x)
      π_loss = -𝔼(qfx-kqlogπx_activ)

      next_dist = fwd(ś)
      next_x = no_grad(sample(*next_dist))
      next_logπx = logp(next_x,*next_dist)
      next_lnax = log_activ(next_x)
      next_logπx_activ = next_logπx+next_lnax
      next_kqlogπx_activ = k*(exp(q_1*next_logπx_activ)-1)
      next_πx = exp(next_logπx)
      next_πx_activ = exp(next_logπx_activ)
      next_e = ent(*next_dist)
      next_ke = k*next_e

      next_qf1 = fwdq1t(ś,activ(next_x))
      next_qf2 = fwdq2t(ś,activ(next_x))
      next_qf = minimum(next_qf1,next_qf2)

      q_target = no_grad(r - r_rms.mean + where(b, 0., γ * (next_qf - next_kqlogπx_activ)))
      q1_loss = .5 * 𝔼(sq(fwdq1(s,activ(a)) - q_target))
      q2_loss = .5 * 𝔼(sq(fwdq2(s,activ(a)) - q_target))

      lnp_activ = logp(a,*dist)+log_activ(a)
      reg_rwd = r - k*(exp(q_1*lnp_activ)-1)

      v_loss = add_n([q1_loss, q2_loss])
      πvgrad = concat([reshape(g,[-1]) for g in\
          grad(π_loss, π_vars)+grad(v_loss, v_vars)],-1)
      vals = stack([π_loss, 𝔼(e), q1_loss, r_rms.mean[0]])

      return πvgrad, vals, reg_rwd
  else:
    @tf.function
    def πvgradvals(s, a, ś, b, r):
      dist = fwd(s)
      e = ent(*dist)
      te = tent(*dist)
      kte = k*te
      x = sample(*dist)
      logπx = logp(x,*dist)
      kqlogπx = k*(exp(q_1*logπx)-1)
      πx = exp(logπx)

      qf1x = fwdq1(s,activ(x))
      qf2x = fwdq2(s,activ(x))

      qfx = minimum(qf1x,qf2x)

      π_loss = -𝔼(rwd(s,x)+qfx+kte)

      next_dist = fwd(ś)
      next_x = no_grad(sample(*next_dist))
      next_logπx = logp(next_x,*next_dist)
      next_kqlogπx = k*(exp(q_1*next_logπx)-1)

      next_qf1 = fwdq1t(ś,activ(next_x))
      next_qf2 = fwdq2t(ś,activ(next_x))
      next_qf = minimum(next_qf1,next_qf2)

      q_target = no_grad(where(b, 0., γ * (rwd(ś,next_x)-r_rms.mean+next_qf - next_kqlogπx)))
      q1_loss = .5 * 𝔼(sq(fwdq1(s,activ(a)) - q_target))
      q2_loss = .5 * 𝔼(sq(fwdq2(s,activ(a)) - q_target))

      reg_rwd = r + kte

      v_loss = add_n([q1_loss, q2_loss])
      πvgrad = concat([reshape(g,[-1]) for g in\
          grad(π_loss, π_vars)+grad(v_loss, v_vars)],-1)
      vals = stack([π_loss, 𝔼(e), q1_loss, r_rms.mean[0]])

      return πvgrad, vals, reg_rwd

