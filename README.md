# A Fast Solver for Interpolating Stochastic Differential Equation Diffusion Models for Speech Restoration


This code implements: B. Lay and T. Gerkmann, "A Fast Solver for Interpolating Stochastic Differential Equation Diffusion Models for Speech Restoration", Interspeech 2026, long paper format.
See here https://arxiv.org/abs/2603.09508
Note that in the code the parameter --lambd is $\kappa$ from the paper.

## How to use the eval.py, which is an inference script for trained interpolating SDEs

Examples

Using proposed iSDE-2S with N discretization points for the reverse SDE (0 <= lambda <= 1) starting at reverse_starting_point. This results in 2*N NFEs.
```
python3 eval.py --test_dir <test_directory> --N <number of diffusion timesteps> --experiments_folder /path/to/base_folder --destination_folder final_folder_name --ckpt /path_to_ckptfolder/last.ckpt
--reverse_starting_point 1.0 --sampler_type isde-2s --format ears --timeit --save_normalized --lambd <value between 0 and 1> --ode_method not_used
```



Using midpoint with N discretization points for the probability-flow ODE starting at reverse_starting_point. This results in 2*N NFEs.

```
python3 eval.py --test_dir <test_directory> --N <number of diffusion timesteps> --experiments_folder /path/to/base_folder --destination_folder final_folder_name --ckpt /path_to_ckptfolder/last.ckpt
--reverse_starting_point 1.0 --sampler_type midpoint --format ears --timeit --save_normalized --lambd 0 --ode_method not_used
```

Using Euler-Maruyama with discretization for the reverse SDE (lambda=1) starting at reverse_starting_point. This results in N NFEs.

```
python3 eval.py --test_dir <test_directory> --N <number of diffusion timesteps> --experiments_folder /path/to/base_folder --destination_folder final_folder_name --ckpt /path_to_ckptfolder/last.ckpt
--reverse_starting_point 1.0 --sampler_type pc --corrector_steps 0 --format ears --timeit --save_normalized --lambd 0 --ode_method not_used
```


Using adaptive RK45 for probability-flor ODE starting at reverse_starting_point. The NFEs are adaptively selected.

```
python3 eval.py --test_dir <test_directory> --N <number of diffusion timesteps> --experiments_folder /path/to/base_folder --destination_folder final_folder_name --ckpt /path_to_ckptfolder/last.ckpt
--reverse_starting_point 1.0 --sampler_type ode --corrector_steps 0 --format ears --timeit --save_normalized --lambd 0 --ode_method RK45
```
