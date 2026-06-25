This code implements: B. Lay and T. Gerkmann, "A Fast Solver for Interpolating Stochastic Differential Equation Diffusion Models for Speech Restoration", Interspeech 2026, long paper format.
See here https://arxiv.org/abs/2603.09508

##########################################################################################################

How to use the eval.py, which is an inference script for trained interpolating SDEs:

Example:

Using midpoint with 2=N for the probability-flow ODE starting at reverse_starting_point. This results in 4 NFEs.

´´´
python3 eval.py --test_dir <test_directory> --N <number of diffusion timesteps> --experiments_folder /path/to/base_folder --destination_folder final_folder_name --ckpt /path_to_ckptfolder/last.ckpt
--reverse_starting_point 1.0 --sampler_type midpoint --format ears --timeit --save_normalized --lambd 0 --ode_method not_used
´´
