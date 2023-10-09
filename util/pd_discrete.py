import numpy as np
from const_discrete import q
import tensorflow as tf
from util.tf_util_discrete import *

pot    = None
breg   = None
rf     = None

q = const(q,f32)
e_tf = const(np.e,f32)
pi_hf = const(0.5*np.pi,f32)

@tf.function
def logp(x, logit):
  lnp = log_softmax(logit)
  return gather(lnp, x)

@tf.function
def ent(logit):
  p = softmax(logit)
  lnp = log_softmax(logit)
  return -rsum(p*lnp)

@tf.function
def negent(logit):
  p = softmax(logit)
  lnp = log_softmax(logit)
  return rsum(p*lnp)

@tf.function
def kl(logit1, logit2):
  p1 = softmax(logit1)
  lnp1 = log_softmax(logit1)
  lnp2 = log_softmax(logit2)
  return rsum(p1*(lnp1 - lnp2))

@tf.function
def rs(x, logit):
  return logp(x, logit)

@tf.function
def tent(logit):
  p = softmax(logit)
  lnp = log_softmax(logit)
  return (1. - rsum(exp(q*lnp)))/(q-1.)

@tf.function
def negtent(logit):
  p = softmax(logit)
  lnp = log_softmax(logit)
  return (rsum(exp(q*lnp)) - 1.)/(q-1.)

@tf.function
def tkl(logit1, logit2):
  p1 = softmax(logit1)
  p2 = softmax(logit2)
  lnp2 = log_softmax(logit2)

  return - tent(logit1) + tent(logit2) +\
      rsum((1. - q * exp((q-1.)*lnp2)) * (p1 - p2))/(q-1.)

@tf.function
def rt(x, logit):
  lnp = log_softmax(logit)
  p = softmax(logit)

  lnpx = gather(lnp, x)
  px = exp(lnpx)
  return - tent(logit) -\
      rsum((1. - q * exp((q-1.)*lnp)) * (- p))/(q-1.) -\
      (1. - q * exp((q-1.)*lnpx))/(q-1.)

@tf.function
def exp_pot(logit):
  p = softmax(logit)
  lnp = log_softmax(logit)
  return rsum(p * exp(p)) - e_tf

@tf.function
def exp_breg(logit1, logit2):
  p1 = softmax(logit1)
  p2 = softmax(logit2)
  return exp_pot(logit1) - exp_pot(logit2) -\
      rsum(((p2+1)*exp(p2)) * (p1 - p2))

@tf.function
def rexp(x, logit):
  lnp = log_softmax(logit)
  p = softmax(logit)

  lnpx = gather(lnp, x)
  px = exp(lnpx)
  return exp_pot(logit) +\
      rsum(((p+1)*exp(p)) * (- p)) +\
      ((px+1)*exp(px))

@tf.function
def cos_pot(logit):
  p = softmax(logit)
  lnp = log_softmax(logit)
  return -rsum(p * cos(pi_hf * p))

@tf.function
def cos_breg(logit1, logit2):
  p1 = softmax(logit1)
  p2 = softmax(logit2)
  return cos_pot(logit1) - cos_pot(logit2) +\
      rsum((cos(pi_hf * p2) - pi_hf * p2 * sin(pi_hf * p2))*(p1-p2))

@tf.function
def rcos(x, logit):
  lnp = log_softmax(logit)
  p = softmax(logit)

  lnpx = gather(lnp, x)
  px = exp(lnpx)
  return cos_pot(logit) -\
      rsum((cos(pi_hf * p) - pi_hf * p * sin(pi_hf*p))*(-p)) -\
      (cos(pi_hf * px) - pi_hf * px * sin(pi_hf*px))

@tf.function
def sin_pot(logit):
  p = softmax(logit)
  lnp = log_softmax(logit)
  return rsum(p * sin(pi_hf * p)) - 1.

@tf.function
def sin_breg(logit1, logit2):
  p1 = softmax(logit1)
  p2 = softmax(logit2)
  return sin_pot(logit1) - sin_pot(logit2) -\
      rsum((sin(pi_hf * p2) + pi_hf * p2 * cos(pi_hf * p2))*(p1-p2))

@tf.function
def rsin(x, logit):
  lnp = log_softmax(logit)
  p = softmax(logit)

  lnpx = gather(lnp, x)
  px = exp(lnpx)
  return sin_pot(logit) +\
      rsum((sin(pi_hf * p) + pi_hf * p * cos(pi_hf*p))*(-p)) +\
      (sin(pi_hf * px) + pi_hf * px * cos(pi_hf*px))

def set_pd(reg_type):
  global pot, breg, rf

  if reg_type == 'shannon':
    pot    = negent
    breg   = kl
    rf     = rs
  elif reg_type == 'tsallis':
    pot    = negtent
    breg   = tkl
    rf     = rt
  elif reg_type == 'exp':
    pot    = exp_pot
    breg   = exp_breg
    rf     = rexp
  elif reg_type == 'cos':
    pot    = cos_pot
    breg   = cos_breg
    rf     = rcos
  elif reg_type == 'sin':
    pot    = sin_pot
    breg   = sin_breg
    rf     = rsin
