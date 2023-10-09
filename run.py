import argparse
import os.path as osp
import numpy as np
from env import Env
from os import path
from util import log
import math
import json

def argparser(args=None):
  prs = argparse.ArgumentParser()
  add = prs.add_argument
  add('--env_id', default='Hopper-v3')
  add('--envE_id', default=None)
  add('--seed', type=int, default=0)
  add('--expr_dir', default='expr')
  add('--pi_step', type=int, default=1000)
  add('--d_step', type=int, default=1000)
  add('--alg', type=str, choices=['sac', 'gail', 'airl',
      'fairl', 'rairl', 'mdirl'], default='gail')
  add('--policy', type=str)
  add('--batch_size', type=int, default=256)
  add('--q', type=float, default=2.)
  add('--k', type=float, default=1.)
  add('--k2', type=float, default=1.)
  add('--reg_lvl', type=int, default=0)
  add('--lr', type=float, default=5e-4)
  add('--alpha', type=float, default=1.)
  add('--alphaT', type=float, default=-1)
  add('--gamma', type=float, default=0.99)
  add('--gp_coeff', type=float, default=1e-2)
  add('--num_steps', type=int, default=int(1e6))
  add('--replay_size', type=int, default=int(5e5))
  add('--record_intvl', type=int, default=5000)
  add('--save_intvl', type=int, default=-1)
  add('--burnin_steps', type=int, default=10000)
  add('--num_demo', type=int, default=1000)
  add('--tag', type=str, default='')
  return prs.parse_args(args=args)

def sci(i):
  absi = np.abs(i)
  if 1 <= absi <= 99 or absi == 0.0: return str(int(i))
  elif 0.1 <= absi < 1: return "{:.1f}".format(i)
  p = int(math.floor(np.log(absi) / np.log(10)))
  return str(int(np.sign(i))*int(round((absi/(10**p)))))+'e'+str(p)

def get_expr_name(args):
  expr_name = f'{args.alg}.'
  expr_name += args.env_id.split('-')[0]
  if args.alg != 'sac' and args.envE_id and args.envE_id != args.env_id:
    expr_name += ('-' + args.envE_id.split("-")[0])
  if args.q == 1.0 or args.q == 2.0:
    q_str = str(int(args.q))
  else:
    q_str = "{:.1f}".format(args.q)
  expr_name += f'.q_{q_str}.k_{sci(args.k)}.k2_{sci(args.k2)}'
  if args.alg == 'mdirl':
    if args.alphaT < 0.:
      args.alphaT = args.alpha
      expr_name += f'.α_{sci(args.alpha)}'
    else:
      expr_name += f'.α0_{sci(args.alpha)}.αT_{sci(args.alphaT)}'
  if args.alg != 'sac':
    expr_name += f".n_{args.num_demo}"
  expr_name += f'.seed_{args.seed}'
  if args.tag != '':
    expr_name += ("."+args.tag)
  return expr_name

def main(args):
  nh = 100

  if "Ant-v3" == args.env_id:
    from env import AntEnv

    env = AntEnv(args.env_id)
    eval_env = AntEnv(args.env_id)
    if args.alg == 'sac' or args.envE_id is None:
      args.envE_id = args.env_id
      envE = env
    else:
      envE = AntEnv(args.envE_id)
    ns = 27
    na = np.prod(env.A.shape)
  else:
    env = Env(args.env_id)
    eval_env = Env(args.env_id)
    if args.alg == 'sac' or args.envE_id is None:
      args.envE_id = args.env_id
      envE = env
    else:
      envE = Env(args.envE_id)
    ns = np.prod(env.S.shape)
    na = np.prod(env.A.shape)

  if na == 1: args.policy = 'diag_gaussian'

  expr_name = get_expr_name(args)
  expr_path = path.join(args.expr_dir, expr_name)

  from const import set_const
  set_const(args.q, args.k, args.k2, args.reg_lvl,
      args.alpha, args.alphaT,
      args.gamma, ns, na, nh, env.A.high, args.gp_coeff)

  from util.rnd import set_seed
  set_seed(args.seed, env, envE)

  from util.rms import init
  init()

  from util.pd import set_pd
  set_pd(args.policy)

  log.configure(expr_path, formats=['stdout', 'csv', 'tb'])

  if args.policy == 'gaussian':
    from nn.pi import GaussianPolicyValue
    π = GaussianPolicyValue()
  elif args.policy == 'diag_gaussian':
    from nn.pi import DiagGaussianPolicyValue
    π =  DiagGaussianPolicyValue()
  elif args.policy == 'diag_gaussian2':
    from nn.pi import DiagGaussianPolicyValue2
    π =  DiagGaussianPolicyValue2()

  from loss import set_πv
  set_πv(π.vars,π.q1_vars,π.q2_vars,
      π.fwd,π.fwdq1,π.fwdq2,π.fwdq1t,π.fwdq2t)

  from data import ReplayBuffer
  if args.alg == 'sac':
    buf = ReplayBuffer(args.replay_size, int(1e4), π, env, ns, na)
    D = expert_dataset = None
  else:
    if args.alg == 'gail':
      from nn.disc import GAIL
      D = GAIL()
    elif args.alg == 'airl':
      from nn.disc import AIRL
      D = AIRL(π)
    elif args.alg == 'fairl':
      from nn.disc import FAIRL
      D = FAIRL()
    elif args.alg == 'rairl':
      if args.policy == 'gaussian':
        from nn.disc import RAIRLGaussian
        D = RAIRLGaussian(π)
      elif args.policy == 'diag_gaussian':
        from nn.disc import RAIRLDiagGaussian
        D = RAIRLDiagGaussian(π)
      elif args.policy == 'diag_gaussian2':
        from nn.disc import RAIRLDiagGaussian2
        D = RAIRLDiagGaussian2(π)
    elif args.alg == 'mdirl':
      if args.policy == 'gaussian':
        from nn.disc import MirrorDescentGaussian
        D = MirrorDescentGaussian(π)
      elif args.policy == 'diag_gaussian':
        from nn.disc import MirrorDescentDiagGaussian
        D = MirrorDescentDiagGaussian(π)
      elif args.policy == 'diag_gaussian2':
        from nn.disc import MirrorDescentDiagGaussian2
        D = MirrorDescentDiagGaussian2(π)
    from loss import set_d
    set_d(D.vars, D.loss, D.rwd)
    buf = ReplayBuffer(args.replay_size, int(1e4), π, env, ns, na, D.intype)

    from data import get_trj
    num_demo = args.num_demo
    env_name = args.envE_id.split("-")[0]
    expert_traj_path = f'data/trj{num_demo}.{env_name}.npz'

    expert_dataset = get_trj(args.env_id, expert_traj_path, D.intype)

  from train import train
  train(args.alg, env, eval_env, π, buf,
      D, expert_dataset, args.batch_size,
      π_step=args.pi_step,
      d_step=args.d_step,
      lr_π=args.lr,
      lr_d=args.lr,
      max_steps=args.num_steps,
      expr_path=expr_path,
      record_intvl=args.record_intvl,
      save_intvl=args.save_intvl,
      burnin_steps=args.burnin_steps)

  log.close()

  param_dict = {}
  for idx, var in enumerate(π.vars):
    param_dict[f'pi_{idx}'] = var.numpy()
  from util.rms import r_rms
  for idx, var in enumerate(r_rms.vars):
    param_dict[f'r_rms_{idx}'] = var.numpy()
  if D is not None:
    for idx, var in enumerate(D.vars):
      param_dict[f'd_{idx}'] = var.numpy()
  param_path = osp.join(expr_path, 'param.npz')
  with open(param_path, 'wb') as f:
    np.savez(f, **param_dict)

if __name__ == '__main__':
  args = argparser()
  main(args)
