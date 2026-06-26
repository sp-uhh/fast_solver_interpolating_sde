# A Fast Solver for Interpolating Stochastic Differential Equation Diffusion Models for Speech Restoration


This code implements: 

B. Lay and T. Gerkmann, "A Fast Solver for Interpolating Stochastic Differential Equation Diffusion Models for Speech Restoration", Interspeech 2026, long paper format.
See here https://arxiv.org/abs/2603.09508


## How to use the eval.py

This is the inference script for solving the reverse process. We assume that a pretrained ckpt is given. The commands for training can be found below.

Examples
Using proposed iSDE-2S-kappa with N discretization points for the reverse SDE (0 <= kappa <= 1) starting at reverse_starting_point. This results in 2*N NFEs.
```
python3 eval.py --test_dir <test_directory> --N <number of diffusion timesteps> --experiments_folder /path/to/base_folder --destination_folder final_folder_name --ckpt /path_to_ckptfolder/last.ckpt
--reverse_starting_point 1.0 --sampler_type isde-2s --format ears --timeit --save_normalized --kappa <value between 0 and 1> --adaptive_ode_method not_used
```



Using midpoint with N discretization points for the probability-flow ODE starting at reverse_starting_point. This results in 2*N NFEs.
```
python3 eval.py --test_dir <test_directory> --N <number of diffusion timesteps> --experiments_folder /path/to/base_folder --destination_folder final_folder_name --ckpt /path_to_ckptfolder/last.ckpt
--reverse_starting_point 1.0 --sampler_type midpoint --format ears --timeit --save_normalized --kappa 0 --ode_method not_used
```

Using Euler-Maruyama with discretization for the reverse SDE (kappaa=1) starting at reverse_starting_point. This results in N NFEs.
```
python3 eval.py --test_dir <test_directory> --N <number of diffusion timesteps> --experiments_folder /path/to/base_folder --destination_folder final_folder_name --ckpt /path_to_ckptfolder/last.ckpt
--reverse_starting_point 1.0 --sampler_type pc --corrector_steps 0 --format ears --timeit --save_normalized --kappa 0 --adaptive_ode_method not_used
```


Using adaptive RK45 for probability-flor ODE starting at reverse_starting_point. The NFEs are adaptively selected. The --adaptive_ode_method parameter is only important when --sampler_type is adaptive. Otherwise, the parameter --adaptive_ode_method can be ignored. 
```
python3 eval.py --test_dir <test_directory> --N <number of diffusion timesteps> --experiments_folder /path/to/base_folder --destination_folder final_folder_name --ckpt /path_to_ckptfolder/last.ckpt
--reverse_starting_point 1.0 --sampler_type ode --corrector_steps 0 --format ears --timeit --save_normalized --kappa 0 --adaptive_ode_method RK45
```


## How to use the train.py
Pretrained checkpoint will be uploaded soon. If you want to train an interpolating SDE (such as fOUVE), then:

```
python train.py --base_dir <path_to_dataset/EARS-WHAM_v2_16k> --batch_size 32 --normalize noisy --backbone ncsnpp_v2 --format ears --fs 16000 --num_frames 128 --hop_length 256 --n_fft 510 --spec_abs_exponent 0.5 --spec_factor 0.1 --audiologs_every_epoch 5 --speclogs_every_epoch 5 --wandb_entity <enter_your_wandb_name> --wandb_project_name <enter_your_project_name> --num_eval_files 5 --save_every_n_epochs 0 --wandb_name <enter_your_wandb_name> --loss_type dsm --sde fouve --gains 0 0 --output_scale sigma --ckpt_destination <path_to_ckpt_folder> --N_inf 60 --loss_weight_type one --t_eps 0.03 --theta 2.0 --sigma-min 0.0001 --sigma-max 0.4
```
The training currently only runs with wandb logging and does not support tensorboard. Alternatively --nolog turns off all wandb parameters (there will be no logging).
