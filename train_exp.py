#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:36:15 2020

@author: mincheol
"""
from baselines import run
import mpi4py

## Block (Position control approach)
defaultargs = ['--alg=her','--env=Block-v1', '--num_timesteps=5e4']
for dim in [4]:
    for seed in [10,500,1000]:
        savepath = '--save_path=./models/block/pos_ctrl_{}'.format(seed)
        demofile = '--demo_file=./block_demo_25.npz'
        logpath = '--log_path=./models/block/pos_ctrl_{}_log'.format(seed)
        perturb = '--perturb=delay'
        algdim = '--algdim={}'.format(dim)
        
        finalargs = defaultargs + [savepath, demofile, logpath, perturb, algdim, '--seed={}'.format(seed)]
        
        run.main(finalargs)
        
# ## Block (SCAPE)
# defaultargs = ['--alg=her','--env=Block-v1', '--num_timesteps=5e4']
# for dim in [6]:
#     for seed in [10,500,1000]:
#         savepath = '--save_path=./models/block/stf_ctrl_{}'.format(seed)
#         demofile = '--demo_file=./block_demo_25_augmented.npz'
#         logpath = '--log_path=./models/block/stf_ctrl_{}_log'.format(seed)
#         perturb = '--perturb=delay'
#         algdim = '--algdim={}'.format(dim)
        
#         finalargs = defaultargs + [savepath, demofile, logpath, perturb, algdim, '--seed={}'.format(seed)]
        
#         run.main(finalargs)

# ## Chip (Position control approach)
# defaultargs = ['--alg=her','--env=Chip-v1', '--num_timesteps=1e5']
# for dim in [3]:
#     for seed in [10,500,1000]:
#         savepath = '--save_path=./models/chip/pos_ctrl_{}'.format(seed)
#         demofile = '--demo_file=./chip_demo_25.npz'
#         logpath = '--log_path=./models/chip/pos_ctrl_{}_log'.format(seed)
#         perturb = '--perturb=delay'
#         algdim = '--algdim={}'.format(dim)
        
#         finalargs = defaultargs + [savepath, demofile, logpath, perturb, algdim, '--seed={}'.format(seed)]
        
#         run.main(finalargs)
        
# ## Chip (SCAPE)
# defaultargs = ['--alg=her','--env=Chip-v1', '--num_timesteps=1e5']
# for dim in [5]:
#     for seed in [10,500,1000]:
#         savepath = '--save_path=./models/chip/stf_ctrl_{}'.format(seed)
#         demofile = '--demo_file=./chip_demo_25_augmented.npz'
#         logpath = '--log_path=./models/chip/stf_ctrl_{}_log'.format(seed)
#         perturb = '--perturb=delay'
#         algdim = '--algdim={}'.format(dim)
        
#         finalargs = defaultargs + [savepath, demofile, logpath, perturb, algdim, '--seed={}'.format(seed)]
        
#         run.main(finalargs)

# ## NuFingers (Position control approach)
# defaultargs = ['--alg=her','--env=NuFingers-v1', '--num_timesteps=1e5']
# for dim in [2]:
#     for seed in [10,500,1000]:
#         savepath = '--save_path=./models/nufingers/pos_ctrl_{}'.format(seed)
#         demofile = '--demo_file=./nufingers_demo_25.npz'
#         logpath = '--log_path=./models/nufingers/pos_ctrl_{}_log'.format(seed)
#         perturb = '--perturb=delay'
#         algdim = '--algdim={}'.format(dim)
        
#         finalargs = defaultargs + [savepath, demofile, logpath, perturb, algdim, '--seed={}'.format(seed)]
        
#         run.main(finalargs)
        
# ## NuFingers (SCAPE)
# defaultargs = ['--alg=her','--env=NuFingers-v1', '--num_timesteps=1e5']
# for dim in [4]:
#     for seed in [10,500,1000]:
#         savepath = '--save_path=./models/nufingers/stf_ctrl_{}'.format(seed)
#         demofile = '--demo_file=./nufingers_demo_25_augmented.npz'
#         logpath = '--log_path=./models/nufingers/stf_ctrl_{}_log'.format(seed)
#         perturb = '--perturb=delay'
#         algdim = '--algdim={}'.format(dim)
        
#         finalargs = defaultargs + [savepath, demofile, logpath, perturb, algdim, '--seed={}'.format(seed)]
        
#         run.main(finalargs)

# ## Hybrid approach (Imitation learning for position control -> reinforcement learning for stiffness control)
# ## Block (first stage)
# defaultargs = ['--alg=her','--env=Block-v1', '--num_timesteps=2.5e4']
# for dim in [4]:
#         savepath = '--save_path=./models/block/hybrid_pos_ctrl'
#         demofile = '--demo_file=./block_demo_25.npz'
#         logpath = '--log_path=./models/block/hybrid_pos_ctrl_log'
#         perturb = '--perturb=none'
#         algdim = '--algdim={}'.format(dim)
        
#         finalargs = defaultargs + [savepath, demofile, logpath, perturb, algdim]
        
#         run.main(finalargs)
        
# ## Block (second stage)
# defaultargs = ['--alg=her','--env=Block-v2', '--num_timesteps=2.5e4']
# for dim in [6]:
#         savepath = '--save_path=./models/block/hybrid_stf_ctrl'
#         logpath = '--log_path=./models/block/hybrid_stf_ctrl_log'
#         perturb = '--perturb=delay'
#         algdim = '--algdim={}'.format(dim)
        
#         finalargs = defaultargs + [savepath, logpath, perturb, algdim]
        
#         run.main(finalargs)
        
# ## Chip (first stage)
# defaultargs = ['--alg=her','--env=Chip-v1', '--num_timesteps=5e4']
# for dim in [3]:
#         savepath = '--save_path=./models/chip/hybrid_pos_ctrl'
#         demofile = '--demo_file=./chip_demo_25.npz'
#         logpath = '--log_path=./models/chip/hybrid_pos_ctrl_log'
#         perturb = '--perturb=none'
#         algdim = '--algdim={}'.format(dim)
        
#         finalargs = defaultargs + [savepath, demofile, logpath, perturb, algdim]
        
#         run.main(finalargs)
        
# ## Chip (second stage)
# defaultargs = ['--alg=her','--env=Chip-v2', '--num_timesteps=5e4']
# for dim in [5]:
#         savepath = '--save_path=./models/chip/hybrid_stf_ctrl'
#         logpath = '--log_path=./models/chip/hybrid_stf_ctrl_log'
#         perturb = '--perturb=delay'
#         algdim = '--algdim={}'.format(dim)
        
#         finalargs = defaultargs + [savepath, logpath, perturb, algdim]
        
#         run.main(finalargs)
        
# ## NuFingers (first stage)
# defaultargs = ['--alg=her','--env=NuFingers-v1', '--num_timesteps=1e5']
# for dim in [2]:
#         savepath = '--save_path=./models/nufingers/hybrid_pos_ctrl'
#         demofile = '--demo_file=./nufingers_demo_25.npz'
#         logpath = '--log_path=./models/nufingers/hybrid_pos_ctrl_log'
#         perturb = '--perturb=none'
#         algdim = '--algdim={}'.format(dim)
        
#         finalargs = defaultargs + [savepath, demofile, logpath, perturb, algdim]
        
#         run.main(finalargs)
        
# ## NuFingers (second stage)
# defaultargs = ['--alg=her','--env=NuFingers-v2', '--num_timesteps=1e5']
# for dim in [4]:
#         savepath = '--save_path=./models/nufingers/hybrid_stf_ctrl'
#         logpath = '--log_path=./models/nufingers/hybrid_stf_ctrl_log'
#         perturb = '--perturb=delay'
#         algdim = '--algdim={}'.format(dim)
        
#         finalargs = defaultargs + [savepath, logpath, perturb, algdim]
        
#         run.main(finalargs)