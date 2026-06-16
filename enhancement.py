import glob
from argparse import ArgumentParser
import os
from os.path import join

import torch
from soundfile import write
from torchaudio import load
from tqdm import tqdm

from sgmse.model import ScoreModel
from sgmse.util.other import ensure_dir, pad_spec

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True, help='Directory containing the test data (must have subdirectory noisy/)')
    parser.add_argument("--enhanced_dir", type=str, required=True, help='Directory containing the enhanced data')
    parser.add_argument("--ckpt", type=str,  help='Path to model checkpoint.')
    parser.add_argument("--corrector", type=str, choices=("ald", "langevin", "none"), default="ald", help="Corrector class for the PC sampler.")
    parser.add_argument("--corrector_steps", type=int, default=1, help="Number of corrector steps")
    parser.add_argument("--snr", type=float, default=0.5, help="SNR value for (annealed) Langevin dynmaics.")
    parser.add_argument("--N", type=int, default=30, help="Number of reverse steps")
    #store ears action True
    parser.add_argument("--ears", action='store_true', help="Store enhanced files in subfolders")
    #set default False
    parser.set_defaults(ears=False)
    parser.add_argument("--rsp", type=float, default=1.0, help="Corrector class for the PC sampler.")

    
    args = parser.parse_args()

    noisy_dir = join(args.test_dir, 'noisy/')
    checkpoint_file = args.ckpt
    corrector_cls = args.corrector
    reverse_starting_point = args.rsp

    target_dir = join(args.enhanced_dir, 'files')
    ensure_dir(target_dir)

    # Settings
    sr = 16000
    snr = args.snr
    N = args.N
    corrector_steps = args.corrector_steps

    # Load score model 
    model = ScoreModel.load_from_checkpoint(checkpoint_file, base_dir='', batch_size=16, num_workers=0, kwargs=dict(gpu=False))
    model.eval(no_ema=False)
    model.cuda()

    if model.sde.__class__.__name__ == 'OUVESDE':
        model.sde._T = reverse_starting_point
    else:  
        model.sde.T = reverse_starting_point
    
    if not args.ears:
        noisy_files = sorted(glob.glob('{}/*.wav'.format(noisy_dir)))
    else:
        noisy_files = sorted(glob.glob('{}/**/*.wav'.format(noisy_dir)))


    for noisy_file in tqdm(noisy_files):
        filename = noisy_file.split('/')[-1]
        
        # Load wav
        y, _ = load(noisy_file) 
        T_orig = y.size(1)   

        # Normalize
        norm_factor = y.abs().max()
        y = y / norm_factor
        
        # Prepare DNN input
        Y = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
        Y = pad_spec(Y)
        
        # Reverse sampling
        if model.loss_type == 'onestep_v2':
            pass
        else:
            x_hat = model.enhance(y, sampler_type='pc', predictor='reverse_diffusion', 
                        corrector=corrector_cls, 
                        corrector_steps=corrector_steps, N=N, snr=snr, 
                        output_scale = 'time',
                        atol=1e-5, rtol=1e-5, timestep_type='linear', correct_stepsize=True)

    

        # Write enhanced wav file
        if not args.ears:
            write(join(target_dir, filename), x_hat.cpu().numpy(), 16000)
        else:
            os.makedirs(target_dir + '/'+noisy_file.split('/')[-2]+ '/', exist_ok=True)
            write(target_dir + '/'+noisy_file.split('/')[-2]+ '/'+filename, x_hat, 16000)

    
    
    # Save settings
    text_file = join(target_dir, "_settings.txt")
    with open(text_file, 'w') as file:
        file.write("checkpoint file: {}\n".format(checkpoint_file))
        file.write("corrector_steps: {}\n".format(corrector_steps))
        file.write("corrector_cls: {}\n".format(corrector_cls))
        file.write("noisy_dir: {}\n".format(noisy_dir))
        file.write("N: {}\n".format(N))
        file.write("snr: {}\n".format(snr))
        file.write("ears: {}\n".format(args.ears))
        file.write("reverse starting point: {}\n".format(args.rsp))

