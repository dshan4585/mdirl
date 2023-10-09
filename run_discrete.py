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
  add('--seed', type=int, default=0)
  add('--expr_dir', default='expr')
  add('--pi_step', type=int, default=1000)
  add('--d_step', type=int, default=1000)
  add('--alg', type=str, choices=['rairl', 'mdirl'], default='rairl')
  add('--reg_type', type=str, choices=['shannon', 'tsallis', 'exp', 'cos', 'sin'], default='shannon')
  add('--batch_size', type=int, default=256)
  add('--q', type=float, default=2.)
  add('--alpha', type=float, default=1.)
  add('--alphaT', type=float, default=-1)
  add('--num_steps', type=int, default=int(5e5))
  add('--num_action', type=int)
  add('--record_intvl', type=int, default=1)
  add('--save_intvl', type=int, default=10000)
  add('--expert_type', type=str, choices=['set1', 'set2', 'random'], default='random')
  add('--burnin_steps', type=int, default=0)
  add('--tag', type=str, default='')
  return prs.parse_args(args=args)

def sci(i):
  absi = np.abs(i)
  if 1 <= absi <= 99 or absi == 0.0: return str(int(i))
  elif 0.1 <= absi < 1: return "{:.1f}".format(i)
  p = int(math.floor(np.log(absi) / np.log(10)))
  return str(int(np.sign(i))*int(round((absi/(10**p)))))+'e'+str(p)

def get_expr_name(args):
  expr_name = f'{args.alg}.{args.reg_type}.{args.num_action}'
  if args.alg == 'mdirl':
    if args.alphaT < 0.:
      args.alphaT = args.alpha
      expr_name += f'.α_{sci(args.alpha)}'
    else:
      expr_name += f'.α0_{sci(args.alpha)}.αT_{sci(args.alphaT)}'
  expr_name += ("."+args.expert_type)
  expr_name += f'.seed_{args.seed}'
  if args.tag != '':
    expr_name += ("."+args.tag)
  return expr_name

def main(args):
  expr_name = get_expr_name(args)
  expr_path = path.join(args.expr_dir, expr_name)

  from const_discrete import set_const
  set_const(args.reg_type, args.q, args.alpha, args.alphaT, args.num_action)

  from util.rnd import set_seed
  from util.pd_discrete import set_pd
  set_pd(args.reg_type)

  log.configure(expr_path, formats=['stdout', 'csv', 'tb'])

  from discrete_models import DiscreteStateless
  π = DiscreteStateless()
  πE = DiscreteStateless()
  if args.expert_type != 'random':
    assert args.num_action == 4
    p1 = np.asarray([.1,.2,.3,.4],dtype=np.float32)
    p2 = np.asarray([.001,.001,.299,.699],dtype=np.float32)

    if args.expert_type == 'set1':
      πE.logit.assign(np.log(p1))
    elif args.expert_type == 'set2':
      πE.logit.assign(np.log(p2))
  else:
    logit = np.random.uniform(low=-10., high=.0, size=args.num_action).astype(np.float32)
    πE.logit.assign(logit)

  if args.alg == 'rairl':
    from discrete_models import RAIRLDiscrete
    D = RAIRLDiscrete(π)
  elif args.alg == 'mdirl':
    from discrete_models import MirrorDescentDiscrete
    D = MirrorDescentDiscrete(π)
  from loss import set_d
  set_d(D.vars, D.loss, D.rwd)

  from train_discrete import train
  train(args.alg, π,πE,
      D, args.batch_size,
      π_step=args.pi_step,
      d_step=args.d_step,
      max_steps=args.num_steps,
      expr_path=expr_path,
      record_intvl=args.record_intvl,
      save_intvl=args.save_intvl,
      burnin_steps=args.burnin_steps)

  log.close()

  param_dict = {}
  for idx, var in enumerate(π.vars):
    param_dict[f'pi_{idx}'] = var.numpy()
  for idx, var in enumerate(πE.vars):
    param_dict[f'piE_{idx}'] = var.numpy()
  if D is not None:
    for idx, var in enumerate(D.vars):
      param_dict[f'd_{idx}'] = var.numpy()
  param_path = osp.join(expr_path, 'param.npz')
  with open(param_path, 'wb') as f:
    np.savez(f, **param_dict)

if __name__ == '__main__':
  args = argparser()
  main(args)
