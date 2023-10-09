import numpy as np
import tensorflow as tf
from const import q, na, a_limit
from util.tf_util import *

logp   = None
prior_logp = None
sample = None
sample_uniform = None
ent    = None
ent2   = None
kl     = None
kl_std = None
tent   = None
tent2  = None
tkl    = None
rs     = None
rt     = None
rs_tan = None
rt_tan = None
dfs    = None
dft    = None
reg    = None

d = const(na,f32)
c = const(.5*np.log(2*np.pi),f32)
ce = const(.5*np.log(2*np.pi*np.e),f32)
dc = const(na*.5*np.log(2*np.pi),f32)
dce = const(na*.5*np.log(2*np.pi*np.e),f32)
dc_neg = const(-na*.5*np.log(2*np.pi),f32)

logp_uniform = const(na*np.log(.5),f32)
p_uniform = const(np.exp(na*np.log(.5)),f32)

ln4 = const(np.log(4),f32)
ln_alim = const(np.log(a_limit),f32)

perm = ed(tf.range(na,dtype=tf.int32),0)

if q > 1.0:
  dq = const(na*.5*np.log(q),f32)
  q_1 = const(q-1,f32)
  q_1_inv = const(1/(q-1),f32)
  qq_1_inv = const(q/(q-1),f32)
  qq_1_inv_neg = const(q/(1-q),f32)
  dcq_1_neg = const(na*(1-q)*.5*np.log(2*np.pi),f32)
  dcq_1_dq_neg = const(na*(1-q)*.5*np.log(2*np.pi)-na*.5*np.log(q),f32)
  q = const(q,f32)
  small_diag = const(tf.eye(na, batch_shape=[1])*1e-1)

@tf.function
def logp_g(x, μ, lnσ, ult):
  return dc_neg-rsum(lnσ+.5*sq(mv(luinv(ult,perm),x-μ))*exp(-2*lnσ),-1)

@tf.function
def logp_d(x, μ, lnσ):
  return dc_neg-rsum(lnσ+.5*sq(x-μ)*exp(-2*lnσ),-1) 

@tf.function
def sample_g(μ, lnσ, ult):
  return μ+mv(ult,exp(lnσ)*normal(shape(μ)))

@tf.function
def sample_d(μ, lnσ):
  return μ+exp(lnσ)*normal(shape(μ))

@tf.function
def sample_uniform_g(μ, lnσ, ult):
  u = uniform(shape(μ), -0.99999, 0.99999)
  return u, atanh(u)

@tf.function
def sample_uniform_d(μ, lnσ):
  u = uniform(shape(μ), -0.99999, 0.99999)
  return u, atanh(u)

@tf.function
def log_activ(x):
  return -rsum(log_sigmoid(2*x)+log_sigmoid(-2*x)+ln4+ln_alim,-1)

@tf.function
def ent_g(μ, lnσ, ult):
  return dce+rsum(lnσ,-1)

@tf.function
def ent_d(μ, lnσ):
  return dce+rsum(lnσ,-1)

@tf.function
def ent2_g(μ, lnσ, ult):
  return rsum(lnσ,-1)

@tf.function
def ent2_d(μ, lnσ):
  return rsum(lnσ,-1)

@tf.function
def kl_g(μ1, lnσ1, ult1, μ2, lnσ2, ult2):
  ult2_inv = luinv(ult2,perm)
  L1 = ult1 * ed(exp(lnσ1),-2)
  L2_inv = ed(exp(-lnσ2),-1) * ult2_inv
  Σ1 = mm(L1, L1, transpose_b=True)
  Σ2_inv = mm(L2_inv, L2_inv, transpose_a=True)
  return .5*(-d+tr(mm(Σ2_inv,Σ1))+\
      rsum(2*(lnσ2-lnσ1)+sq(mv(ult2_inv,μ2-μ1))*exp(-2*lnσ2),-1))

@tf.function
def kl_d(μ1, lnσ1, μ2, lnσ2):
  return .5*(-d+rsum(2*(lnσ2-lnσ1)+exp(2*(lnσ1-lnσ2))+sq(μ2-μ1)*exp(-2*lnσ2),-1))

@tf.function
def kl_std_g(μ, lnσ, ult):
  L_sq = sq(ult) * ed(exp(2*lnσ),-2)
  trΣ = rsum(L_sq, [-2,-1])
  return .5*(-d+trΣ+rsum(-2*lnσ+sq(μ),-1))

@tf.function
def kl_std_d(μ, lnσ):
  return .5*(-d+rsum(-2*lnσ+exp(2*lnσ)+sq(μ),-1))

@tf.function
def tent_g(μ, lnσ, ult):
  return q_1_inv*(1-exp(dcq_1_dq_neg-q_1*rsum(lnσ,-1)))

@tf.function
def tent_d(μ, lnσ):
  return q_1_inv*(1-exp(dcq_1_dq_neg-q_1*rsum(lnσ,-1)))

@tf.function
def tent2_g(μ, lnσ, ult):
  return q_1_inv*(1-exp(-q_1*rsum(lnσ,-1)))

@tf.function
def tent2_d(μ, lnσ):
  return q_1_inv*(1-exp(-q_1*rsum(lnσ,-1)))

@tf.function
def tkl_g(μ1, lnσ1, ult1, μ2, lnσ2, ult2):
  ult1_inv = luinv(ult1,perm)
  ult2_inv = luinv(ult2,perm)
  L1_inv = ed(exp(-lnσ1),-1) * ult1_inv
  L2_inv = ed(exp(-lnσ2),-1) * ult2_inv
  Σ1_inv = mm(L1_inv, L1_inv, transpose_a=True)
  Σ2_inv = mm(L2_inv, L2_inv, transpose_a=True)
  a = mv(Σ1_inv, μ1) + q_1 * mv(Σ2_inv, μ2)
  B = Σ1_inv + q_1 * Σ2_inv
  return maximum(qq_1_inv_neg*exp(dcq_1_neg - .5*logdet(B) + rsum(
      .5 * a * mv(inv(B), a) -\
      .5 * sq(mv(ult1_inv, μ1))*exp(-2*lnσ1) - lnσ1 - q_1 * (
      .5 * sq(mv(ult2_inv, μ2))*exp(-2*lnσ2) + lnσ2), -1))+\
      q_1_inv*exp(dcq_1_dq_neg-q_1*rsum(lnσ1,-1))+\
      exp(dcq_1_dq_neg-q_1*rsum(lnσ2,-1)),0.)

@tf.function
def tkl_d(μ1, lnσ1, μ2, lnσ2):
  σ1_inv2 = exp(-2 * lnσ1)
  σ2_inv2 = exp(-2 * lnσ2)
  b = σ1_inv2 +  q_1 * σ2_inv2

  return qq_1_inv_neg*exp(dcq_1_neg + rsum(
      .5 * sq(μ1 * σ1_inv2 + q_1 * μ2 * σ2_inv2) / b -\
      .5 * ln(b) -\
      .5 * sq(μ1)*σ1_inv2 - lnσ1 - q_1 * (
      .5 * sq(μ2)*σ2_inv2 + lnσ2), -1))+\
      q_1_inv*exp(dcq_1_dq_neg-q_1*rsum(lnσ1,-1))+\
      exp(dcq_1_dq_neg-q_1*rsum(lnσ2,-1))

@tf.function
def rs_g(x, μ, lnσ, ult):
  return dc_neg-rsum(lnσ+.5*sq(mv(luinv(ult,perm),x-μ))*exp(-2*lnσ),-1)

@tf.function
def rs_d(x, μ, lnσ):
  return dc_neg-rsum(lnσ+.5*sq(x-μ)*exp(-2*lnσ),-1)

@tf.function
def rt_g(x, μ, lnσ, ult):
  return qq_1_inv*exp(dcq_1_neg-q_1*rsum(lnσ+.5*sq(mv(luinv(ult,perm),x-μ))*exp(-2*lnσ),-1)) - exp(dcq_1_dq_neg-q_1*rsum(lnσ,-1))-q_1_inv

@tf.function
def rt_d(x, μ, lnσ):
  return qq_1_inv*exp(dcq_1_neg-q_1*rsum(lnσ+.5*sq(x-μ)*exp(-2*lnσ),-1)) - exp(dcq_1_dq_neg-q_1*rsum(lnσ,-1))-q_1_inv

@tf.function
def bound_scale_g(x_μ, lnσ, ult, ε):
  return sqrt(relu(dc_neg - ln(ε) - rsum(lnσ,-1))/ maximum(rsum(.5*sq(mv(luinv(ult,perm),x_μ))*exp(-2*lnσ),-1),1e-8))

@tf.function
def bound_scale_d(x_μ, lnσ, ε):
  return sqrt(relu(dc_neg - ln(ε) - rsum(lnσ,-1))/ maximum(rsum(.5*sq(x_μ)*exp(-2*lnσ),-1),1e-8))

@tf.function
def rs_tan_g(x, μ, lnσ, ult, ε=1e-2):
  x_μ = x - μ
  scale = bound_scale_g(x_μ, lnσ, ult, ε)
  r = rs_g(x, μ, lnσ, ult)
  x_bnd = ed(scale,-1) * x_μ + μ
  r_bnd = rs_g(x_bnd, μ, lnσ, ult)
  g = grad(r_bnd, [x_bnd])[0]
  r_lin = rsum(g * (x - x_bnd),-1) + r_bnd
  return tf.where(greater(scale, 1.), r, r_lin)

@tf.function
def rs_tan_d(x, μ, lnσ, ε=1e-2):
  x_μ = x - μ
  scale = bound_scale_d(x_μ, lnσ, ε)
  r = rs_d(x, μ, lnσ)
  x_bnd = ed(scale,-1) * x_μ + μ
  r_bnd = rs_d(x_bnd, μ, lnσ)
  g = grad(r_bnd, [x_bnd])[0]
  r_lin = rsum(g * (x - x_bnd),-1) + r_bnd
  return tf.where(greater(scale, 1.), r, r_lin)

@tf.function
def rt_tan_g(x, μ, lnσ, ult, ε=1e-2):
  x_μ = x - μ
  scale = bound_scale_g(x_μ, lnσ, ult, ε)
  r = rt_g(x, μ, lnσ, ult)
  x_bnd = ed(scale,-1) * x_μ + μ
  r_bnd = rt_g(x_bnd, μ, lnσ, ult)
  g = grad(r_bnd, [x_bnd])[0]
  r_lin = rsum(g * (x - x_bnd),-1) + r_bnd
  return tf.where(greater(scale, 1.), r, r_lin)

@tf.function
def rt_tan_d(x, μ, lnσ, ε=1e-2):
  x_μ = x - μ
  scale = bound_scale_d(x_μ, lnσ, ε)
  r = rt_d(x, μ, lnσ)
  x_bnd = ed(scale,-1) * x_μ + μ
  r_bnd = rt_g(x_bnd, μ, lnσ, ult)
  g = grad(r_bnd, [x_bnd])[0]
  r_lin = rsum(g * (x - x_bnd),-1) + r_bnd
  return tf.where(greater(scale, 1.), r, r_lin)

_x_bnd = np.zeros((1,na),np.float64)
_x_bnd[:,0] = np.sqrt(2.*(-na*.5*np.log(2*np.pi)-np.log(1e-8)))
x_bnd = const(_x_bnd,f32)

rs_min = rs_d(x_bnd, zeros((1,na)), zeros((1,na)))
if q == 1.0:
  rt_min = rs_min
else:
  rt_min = rt_d(x_bnd, zeros((1,na)), zeros((1,na)))

@tf.function
def dfs_g(x, μ, lnσ, ult):
  return dc_neg-rsum(lnσ+.5*sq(mv(luinv(ult,perm),x-μ))*exp(-2*lnσ),-1)+1

@tf.function
def dfs_d(x, μ, lnσ):
  return dc_neg-rsum(lnσ+.5*sq(x-μ)*exp(-2*lnσ),-1)+1

@tf.function
def dft_g(x, μ, lnσ, ult):
  return qq_1_inv*exp(dcq_1_neg-q_1*rsum(lnσ+.5*sq(mv(luinv(ult,perm),x-μ))*exp(-2*lnσ),-1)) -q_1_inv

@tf.function
def dft_d(x, μ, lnσ):
  return qq_1_inv*exp(dcq_1_neg-q_1*rsum(lnσ+.5*sq(x-μ)*exp(-2*lnσ),-1)) - q_1_inv

@tf.function
def reg_g(μ, lnσ, ult):
  return 0.5 * 𝔼(μ) + 0.5 * 𝔼(μ) + 0.5 * 𝔼(ult)

@tf.function
def reg_d(μ, lnσ):
  return 0.5 * 𝔼(μ) + 0.5 * 𝔼(μ)

def set_pd(policy_type):
  global logp,sample,sample_uniform,ent,ent2,kl,kl_std,tent,tent2,tkl,rs,rt,rs_tan,rt_tan,dfs,dft,reg
  if policy_type == 'gaussian':
    logp           = logp_g
    sample         = sample_g
    sample_uniform = sample_uniform_g
    ent            = ent_g
    ent2           = ent2_g
    kl             = kl_g
    kl_std         = kl_std_g
    tent           = tent_g
    tent2          = tent2_g
    tkl            = tkl_g
    rs             = rs_g
    rt             = rt_g
    rs_tan         = rs_tan_g
    rt_tan         = rt_tan_g
    dfs            = dfs_g
    dft            = dft_g
    reg            = reg_g
  else:
    logp           = logp_d
    sample         = sample_d
    sample_uniform = sample_uniform_d
    ent            = ent_d
    ent2           = ent2_d
    kl             = kl_d
    kl_std         = kl_std_d
    tent           = tent_d
    tent2          = tent2_d
    tkl            = tkl_d
    rs             = rs_d
    rt             = rt_d
    rs_tan         = rs_tan_d
    rt_tan         = rt_tan_d
    dfs            = dfs_d
    dft            = dft_d
    reg            = reg_d
