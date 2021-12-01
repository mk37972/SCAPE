# SCAPE
Learning Stiffness Control from Augmented Position control Experiences (SCAPE)

[Link to arXiv](https://arxiv.org/abs/2102.08442)

[Link to OpenReview](https://openreview.net/forum?id=L0tXWRrB9yw)

## Requirements
1. mpi4py
2. tensorflow + tensorflow-determinism (if reproducibility is needed)

## Instructions
Uncomment the desired experiment in train_exp.py.

For example, to run Block (Position control approach), uncomment only:
```
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
```
Run train_exp.py to train the agent.

## Notes
The results in the paper are produced using 8 processors in parallel. (i.e., mpiexec -n 8 python train_exp.py)

The hybrid approach requires the user to run the first stage first, and then move onto the second stage.

If you find this code useful, please consider citing:
```
@article{kim2021scape,
  title={SCAPE: Learning Stiffness Control from Augmented Position Control Experiences},
  author={Kim, Mincheol and Niekum, Scott and Deshpande, Ashish D},
  journal={arXiv preprint arXiv:2102.08442},
  year={2021}
}
```
