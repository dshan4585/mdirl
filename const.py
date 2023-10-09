import numpy as np

q=None
k=None
k2=None
reg_lvl=None
α0=None
αT=None
γ=None
ns=None
na=None
nh=None
u_scale=None
u_limit=None
a_limit=None
a_scale=None
a_limit_scale=None
gp_coeff=None

def set_const(tsq, tsk, tsk2, reg_level, alpha0, alphaT, gamma, num_state, num_action, num_hid, a_high,grad_pen):
  global q, k, k2, reg_lvl, α0, αT, γ, ns, na, nh, u_scale, u_limit, a_limit, a_scale,a_limit_scale,gp_coeff
  q = tsq
  k = tsk
  k2 = tsk2
  reg_lvl = reg_level
  α0 = alpha0
  αT = alphaT
  γ = gamma
  ns = num_state
  na = num_action
  nh = num_hid
  u_limit = np.arctanh(1/1.01)/0.5
  u_scale = 0.5
  a_limit = 1.01*np.ones_like(a_high)
  a_scale = a_high
  a_limit_scale = a_limit * a_scale
  gp_coeff = grad_pen


