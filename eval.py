import numpy as np
import os
import glob
from soundfile import read, write
from tqdm import tqdm
from pesq import pesq
from torchaudio import load

import torch
import distillmos

from argparse import ArgumentParser
from os.path import join
from compute_fadtk import FADTK_embedding_cacher

import pandas as pd
from sgmse.model import ScoreModel
from wvmos import get_wvmos
from pesq import pesq
from pystoi import stoi
from sgmse.util.other import pad_spec
from utils import log_spectral_distance



# backup the original torch.load
original_torch_load = torch.load

# patch torch.load globally
def patched_torch_load(*args, **kwargs):
    # force weights_only=False for all loads
    kwargs.setdefault('weights_only', False)
    return original_torch_load(*args, **kwargs)

torch.load = patched_torch_load


def wvmos_from_signal(signal, wvmos_model, sr=16000):
    x = wvmos_model.processor(signal, return_tensors="pt", padding=True, sampling_rate=16000).input_values
    with torch.no_grad():
        if wvmos_model.cuda_flag:
            x = x.cuda()
        res = wvmos_model.forward(x).mean()
    return res.cpu().item()



from utils import energy_ratios, ensure_dir, print_mean_std

def log_spectra(x, x_hat):
    X_stft = model._stft(torch.tensor(x).cuda())
    Xhat_stft = model._stft(torch.tensor(x_hat).cuda())
    diff = (torch.log(torch.abs(X_stft)) - torch.log(torch.abs(Xhat_stft)))**2
    L2 = torch.sqrt(torch.sum(diff))/diff.shape[1]
    log_spectra = L2.detach().cpu().numpy()
    return float(log_spectra)
    



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--destination_folder", type=str, help="Basefolder to save experiments")
    parser.add_argument("--experiments_folder", type=str, required=True, help='folder name to save experiments')
    parser.add_argument("--test_dir", type=str, required=True, help='Directory containing the test data')
    parser.add_argument("--ckpt", type=str, help='Path to model checkpoint.')
    parser.add_argument("--normalize_by", type=str, default='noisy', help='Normalize input by clean/noisy/none.')
    parser.add_argument("--format", type=str, default='ears', help='Test folder structure.')
    parser.add_argument("--save_num_files", type=int, default=-1, help="How many files to save. -1 for all.")

    parser.add_argument("--save_normalized", action="store_true", help="Save normalized enhanced files.")
    parser.add_argument("--sampler_type", type=str,  default="pc",help="Specify the sampler type")
    parser.add_argument("--adaptive_ode_method", type=str, default='RK45', help="Only used, when sampler_type = adaptive")
    
    parser.add_argument("--predictor", type=str, default="reverse_diffusion", help="Predictor class when sampler_type = PC.")
    parser.add_argument("--corrector", type=str, choices=("ald", "none"), default="ald", help="Corrector class when sampler_type = PC.")
    parser.add_argument("--corrector_steps", type=int, default=1, help="Number of corrector steps")
    parser.add_argument("--snr", type=float, default=0.5, help="SNR value for annealed Langevin dynmaics as the corrector.")
    parser.add_argument("--timeit", action="store_true", default=False, help="Time computations")

    parser.add_argument("--reverse_starting_point", type=float, default=1.0, help="Starting point for the reverse SDE.")
    parser.add_argument("--N", type=int, default=30, help="Number of reverse steps")
    parser.add_argument("--kappa", type=float, default=0, help="1.0 = SDE based sampler, 0 = ODE based sampler")

    parser.add_argument("--probability_flow", action="store_true", default = False, help="Only relevant when sampler_type = PC. Whether to use probability flow ODE or the reverse SDE for the predictor.")
    parser.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance for the ODE sampler")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance for the ODE sampler")
    parser.add_argument("--timestep_type", type=str, default='default', help="timestep for sampling")

    
    args = parser.parse_args()

    clean_dir = join(args.test_dir, "test", "clean")
    noisy_dir = join(args.test_dir, "test", "noisy")

    checkpoint_file = args.ckpt

    target_dir = args.experiments_folder + "/{}/".format(
        args.destination_folder)

    ensure_dir(target_dir + "files/")

    # Settings
    sr = 16000
    sampler_type = args.sampler_type
    N = args.N

    predictor = args.predictor
    timestep_type = args.timestep_type
    corrector = args.corrector
    corrector_steps = args.corrector_steps
    snr = args.snr
    reverse_starting_point = args.reverse_starting_point
    atol = args.atol
    rtol = args.rtol
    
    save_num_files = args.save_num_files

    total_nfe = 0
    total_time = 0.0
    
    wvmos_model = get_wvmos(cuda=True)
    
    sqa_model = distillmos.ConvTransformerSQAModel()
    sqa_model = sqa_model.to('cuda')
    sqa_model.eval()

    # Load score model
    model = ScoreModel.load_from_checkpoint(
        checkpoint_file, base_dir="",
        batch_size=16, num_workers=0, kwargs=dict(gpu=False)
    )
    model.eval(no_ema=False)
    model.cuda()
    if args.format == 'ears':
        noisy_files = sorted(glob.glob('{}/**/*.wav'.format(noisy_dir)))
    elif args.format == 'ears-mp3':
        noisy_files = sorted(glob.glob('{}/**/*.mp3'.format(noisy_dir)))[:2]
    else:
        noisy_files = sorted(glob.glob('{}/*.wav'.format(noisy_dir)))
        

    

    fadtk_cacher = FADTK_embedding_cacher(clean_dir, join(args.test_dir, "test",))

    if save_num_files == -1:
        save_num_files = len(noisy_files)
    
    if model.sde.__class__.__name__ in ['OUVESDE', 'fOUVESDE']:
        model.sde._T = reverse_starting_point
    else:  
        model.sde.T = reverse_starting_point

    data = {"filename": [], "pesq": [], "estoi": [],
            "si_sdr": [], "si_sir": [], "si_sar": [], 
            "WVMOS": [], "LSD": [], "DistillMOS": []}
    for cnt, noisy_file in tqdm(enumerate(noisy_files)):
        if args.format == 'ears' or args.format == 'ears-mp3':
            filename = '/'.join(noisy_file.split('/')[-2:])
        else:
            filename = noisy_file.split('/')[-1]
            
        if args.format == 'ears-mp3':
            filename = filename.replace('.mp3', '.wav')
        
        # Load wav
        x, _ = load(join(clean_dir, filename))
        y, _ = load(noisy_file)
        
        
        #requires only for BWE as the dataset has different length of clean and noisy files
        if x.shape[1] != y.shape[1]:
            _len = min(x.shape[1], y.shape[1])
            x = x[:, :_len]
            y = y[:, :_len]
            
        # normalize w.r.t to the noisy or the clean signal or not at all
        # to ensure same clean signal power in x and y.
        if args.normalize_by == "noisy":
            normfac = y.abs().max()
        elif args.normalize_by == "clean":
            normfac = x.abs().max()
        elif args.normalize_by == "not":
            normfac = 1.0
        y = y / normfac
        


        x_hat = model.enhance(y, sampler_type=sampler_type, predictor=predictor, kappa=args.kappa,
                corrector=corrector, corrector_steps=corrector_steps, N=N, snr=snr,
                atol=atol, rtol=rtol, timestep_type=timestep_type, probability_flow=args.probability_flow,
                timeit=args.timeit, adaptive_ode_method=args.adaptive_ode_method)
        if args.timeit:
            x_hat, nfe, time_elapsed = x_hat
            total_nfe += nfe
            total_time += time_elapsed


        if args.save_normalized:
            x_hat = x_hat / np.max(np.abs(x_hat)) 
        
        # Convert to numpy
        x = x.squeeze().cpu().numpy()
        y = y.squeeze().cpu().numpy()
        n = y - x

        # Write enhanced wav file
        if cnt < save_num_files:
            if args.format == 'ears' or args.format == 'ears-mp3':
                os.makedirs(target_dir + "files/"+ filename.split('/')[0], exist_ok=True)
                write(target_dir + "files/" + filename, x_hat, 16000)
            else:
                write(target_dir + "files/" + filename, x_hat, 16000)

        x_hat_tensor = torch.tensor(x_hat).to('cuda')
        data["filename"].append(filename)
        try:
            p = pesq(sr, x, x_hat, 'wb')
        except: 
            p = float("nan")
        lsd_norm = log_spectral_distance(torch.tensor(x_hat), torch.tensor(x))
        data["LSD"].append(lsd_norm.item())
        data["pesq"].append(p)
        data["estoi"].append(stoi(x, x_hat, sr, extended=True))
        data["si_sdr"].append(energy_ratios(x_hat, x, n)[0])
        data["si_sir"].append(energy_ratios(x_hat, x, n)[1])
        data["si_sar"].append(energy_ratios(x_hat, x, n)[2])
        with torch.no_grad():
            mos = sqa_model(x_hat_tensor.unsqueeze(0))
        data["DistillMOS"].append(float(mos.detach().cpu().numpy()))
        fadtk_cacher.compute_embd_delay(x_hat_tensor)
        wvmos = wvmos_from_signal(x_hat_tensor, wvmos_model, sr=16000)
        if wvmos < 1:
            #Note that wvmos can be smaller than 1, but minimal MOS value is 1.
            wvmos = 1
        data["WVMOS"].append(wvmos)

    # Save results as DataFrame
    df = pd.DataFrame(data)
    df.to_csv(join(target_dir, "_results.csv"), index=False)
    fad_value = fadtk_cacher.compute_fad_value()
    # Save average results
    text_file = join(target_dir, "_avg_results.txt")
    with open(text_file, 'w') as file:
        file.write("PESQ: {} \n".format(print_mean_std(data["pesq"])))
        file.write("ESTOI: {} \n".format(print_mean_std(data["estoi"])))
        file.write("SI-SDR: {} \n".format(print_mean_std(data["si_sdr"])))
        file.write("SI-SIR: {} \n".format(print_mean_std(data["si_sir"])))
        file.write("SI-SAR: {} \n".format(print_mean_std(data["si_sar"])))
        file.write("WVMOS: {} \n".format(print_mean_std(data["WVMOS"])))
        file.write("LSD: {} \n".format(print_mean_std(data["LSD"])))
        file.write("Distill MOS: {} \n".format(print_mean_std(data["DistillMOS"])))
        file.write("FADTK: {} \n".format(fad_value))

    # Save settings
    text_file = join(target_dir, "_settings.txt")
    with open(text_file, 'w') as file:
        file.write("checkpoint file: {}\n".format(checkpoint_file))
        file.write("sampler_type: {}\n".format(sampler_type))
        file.write("save_num_files: {}\n".format(save_num_files))
        file.write("predictor: {}\n".format(predictor))
        file.write("corrector: {}\n".format(corrector))
        file.write("corrector_steps: {}\n".format(corrector_steps))
        file.write("probability_flow: {}\n".format(args.probability_flow))
        file.write("N: {}\n".format(N))
        file.write("format : {}\n".format(args.format))
        file.write("Reverse starting poi nt: {}\n".format(reverse_starting_point))
        file.write("snr: {}\n".format(snr))
        file.write("timestep type: {}\n".format(timestep_type))
        file.write("kappa: {}\n".format(args.kappa))
        file.write("normalize_by: {}\n".format(args.normalize_by))
        file.write("save_normalized: {}\n".format(args.save_normalized))
        file.write("loss type: {}\n".format(model.loss_type))
        file.write("output scale: {}\n".format(model.output_scale))
        file.write("num noisy_files: {}\n".format(cnt+1))
        if args.timeit:
            file.write("timeit: Avg. NFE {}, total time {}, \n".format(
                total_nfe/(cnt+1), total_time))
        if sampler_type == "ode":
            file.write("atol: {}\n".format(atol))
            file.write("rtol: {}\n".format(rtol))
            file.write("adaptive ode method: {}\n".format(args.adaptive_ode_method))
