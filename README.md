# Mirror Descent Inverse Reinforcement Learning (NeurIPS 2022)

This repository contains an implementation of our paper:

> Robust Imitation via Mirror Descent Inverse Reinforcement Learning
> 
> Dong-Sig Han, Hyunseo Kim, Hyundo Lee, Je-Hwan Ryu, and Byoung-Tak Zhang
> 
> Advances in Neural Information Processing Systems 35 (NeurIPS 2022)
> 
> arXiv: [https://arxiv.org/abs/2211.02291](https://arxiv.org/abs/2210.11201)


This implementation makes use of [Tensorflow 2.2.0](https://github.com/tensorflow/tensorflow) and Gym 0.17.2.

The RL loop is a variant of the SAC algorithm in [stable-baselines](https://github.com/hill-a/stable-baselines) to serve our specific purposes of implementing regularized actor-critic and modeling Gaussian policies.

Additionally, for the demonstration code in the `demo_julia` directory, we used [Julia 1.7](https://julialang.org) for training and visualization.

To run the MuJoCo experiments, one needs to place optimal expert trajectory files in the `data` directory (specified with `f'data/trj{num_demo}.{env_name}.npz'`), which then can be stored in memory with `data/trj.py` file.

## Training

Run MD-AIRL (Tsallis entropy, q=2) on MuJoCo
```
python3 run.py --env_id Hopper-v3 --policy gaussian --alg mdirl --q 2 --k 1e-2 --k2 1. --alphaT 20. --num_demo 100 --gamma 0.99 --gp_coeff 1e-4 --burnin_steps 10000 --save_intvl 100000 --seed 0 --num_steps 1000000
```

Run MD-AIRL (Tsallis entropy, q=2) on an discete environment (a.k.a. multi-armed bandits)
```
python3 run_discrete.py --alg mdirl --alpha 1. --alphaT 2. --reg_type tsallis --batch_size 4 --num_action 4 --expert_type set1 --seed 0
```

Run MD-AIRL on noisy MuJoCo
```
python3 run_noisy.py --env_id Hopper-v3 --policy gaussian --alg mdirl --q 2 --noise_lvl 1e-2 --k 1e-2 --k2 1. --alphaT 20. -num_demo 100 --gamma 0.99 --gp_coeff 1e-4 --burnin_steps 10000 --save_intvl 100000 --seed 0 --num_steps 1000000
```

# Reference

~~~
@inproceedings{NEURIPS2022_c1f7b1ed,
 author = {Han, Dong-Sig and Kim, Hyunseo and Lee, Hyundo and Ryu, JeHwan and Zhang, Byoung-Tak},
 title = {Robust Imitation via Mirror Descent Inverse Reinforcement Learning},
 booktitle = {Advances in Neural Information Processing Systems},
 pages = {30031--30043},
 volume = {35},
 year = {2022}
}
~~~
