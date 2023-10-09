import numpy as np

reg_type=None
q=None
α0=None
αT=None
na=None

def set_const(regularization_type, tsq, alpha0, alphaT, num_action):
  global reg_type, q, α0, αT, na
  reg_type=regularization_type
  q=tsq
  α0 = alpha0
  αT = alphaT
  na = num_action