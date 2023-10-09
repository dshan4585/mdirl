π_vars = None
v_vars = None
πv_vars = None
fwd = None
fwdq1 = None
fwdq2 = None
fwdq1t = None
fwdq2t = None
act = None

def set_πv(π_variables, q1_variables, q2_variables, 
    π_fwd, π_fwdq1, π_fwdq2, π_fwdq1t, π_fwdq2t):
  global π_vars, v_vars, πv_vars, fwd, fwdq1, fwdq2, fwdq1t, fwdq2, fwdq2t
  π_vars = π_variables
  v_vars =  q1_variables +\
            q2_variables
  πv_vars = π_variables +\
            q1_variables +\
            q2_variables
  fwd = π_fwd
  fwdq1 = π_fwdq1
  fwdq2 = π_fwdq2
  fwdq1t = π_fwdq1t
  fwdq2t = π_fwdq2t

d_vars = None
d_fn = None
rwd = None

def set_d(d_variables, d_loss_grad_fn, d_rwd_fn):
  global d_vars, d_fn, rwd
  d_vars = d_variables
  d_fn = d_loss_grad_fn
  rwd = d_rwd_fn

