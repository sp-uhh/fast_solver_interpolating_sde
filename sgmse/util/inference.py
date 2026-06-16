import torch
import torch.nn.functional as F
from pesq import pesq
from torchaudio import load
import librosa
from pystoi import stoi
import numpy as np
import torchaudio
import librosa
from .other import si_sdr, pad_spec

# Settings
sr = 16000
snr = 0.5
corrector_steps = 0


def evaluate_model(model, num_eval_files, spec=True, audio=True):
    N = model.N_inf
    clean_files = model.data_module.valid_set.clean_files
    noisy_files = model.data_module.valid_set.noisy_files

    fs = model.data_module.fs
    total_num_files = len(clean_files)
    indices = torch.linspace(0, total_num_files-1, num_eval_files, dtype=torch.int)
    clean_files = list(clean_files[i] for i in indices)
    noisy_files = list(noisy_files[i] for i in indices)
    spec_list = {"y": [], "x_hat": [], "x": [], "fn": []} if spec else None      
    audio_list = {"y": [], "x_hat": [], "x": [], "fn": []} if audio else None   
    basic_metrics = {"si_sdr": np.zeros(num_eval_files), "estoi": np.zeros(num_eval_files), "pesq": np.zeros(num_eval_files)}
        

    # iterate over files
    i = 0
    for (clean_file, noisy_file) in zip(clean_files, noisy_files):
        # Load wavs
        x, fs = load(clean_file)
        y, fs = load(noisy_file)
        fn = noisy_file.split('/')[-1]
        
        
        if x.shape != y.shape:
            _min = np.min([x.shape[-1], y.shape[-1]])
            x = x[:, :_min]
            y = y[:, :_min]

        
        if fs != sr:
            x = torchaudio.functional.resample(x, fs, sr)
            y = torchaudio.functional.resample(y, fs, sr)
         
        T_orig = x.size(1)   
        #sr = fs

        # Normalize per utterance
        if model.data_module.normalize == "noisy":
            normfac = y.abs().max()
        elif model.data_module.normalize == "clean":
            normfac = x.abs().max()
        elif model.data_module.normalize == "not":
            normfac = 1.0
            
        y = y / normfac

        # Prepare DNN input
        Y = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
        Y = pad_spec(Y)
        y = y * normfac

        if model.loss_type == 'onestep':
            Xt, _ = model.sde.prior_sampling(Y.shape,Y)
            vec_t = torch.ones(Y.shape[0], device=Y.device) * model.sde.T
            sample = model.forward(Xt, vec_t, Y, vec_t[:,None,None,None])
        elif model.loss_type == 'onestep_v2':
            Xt, _ = model.sde.prior_sampling(Y.shape,Y)
            vec_t = torch.ones(Y.shape[0], device=Y.device) * model.sde.T
            sample = model.forward(Xt, vec_t, Y, vec_t[:,None,None,None])
            sample = Xt - sample
        elif model.loss_type in ['ot', 'ot_time']:
            Xt, _ = model.sde.prior_sampling(Y.shape,Y)

            timesteps = torch.linspace(model.sde.T, 0, N+1)
            for ii in range(len(timesteps) - 1):
                t = timesteps[ii]
                
                std = model.sde._std(t).unsqueeze(0)
                sigmas = std[:, None, None, None]
                
                if model.output_scale=='time':
                    scale = t[:,None, None, None]
                elif model.output_scale=='sigma':
                    scale = sigmas
                elif model.output_scale=='no':
                    scale = torch.ones_like(sigmas)
                else:
                    raise ValueError('output scale not implemented!')
                
                
                if ii != len(timesteps) - 1:
                    stepsize = t - timesteps[ii+1]
                vec_t = torch.ones(Y.shape[0], device=Y.device) * t
                vectfield = model.forward(Xt, vec_t, Y, scale.to(device=Y.device))
                Xt = Xt - vectfield*stepsize
            sample = Xt
        elif model.loss_type == 'dp':
            Xt, _ = model.sde.prior_sampling(Y.shape,Y)
            timesteps = torch.linspace(model.sde.T, 0, N+1)
            for ii in range(len(timesteps)):
                t = timesteps[ii]
                if ii == len(timesteps) - 1:
                    break
                vec_t = torch.ones(Y.shape[0], device=Y.device) * t
                scale = model.get_outputscale(vec_t) 
                xhat = model.forward(Xt, vec_t, Y, scale)
                _mean, _std = model.sde.marginal_prob(xhat, vec_t, Y)
                _std = _std.to(device=Y.device)
                Xt = _mean + _std*torch.randn_like(Xt)
            sample = Xt    
        
        else:
            rsp = model.sde.T
            # PC sampler
            
            sampler = model.get_sampler(
                'reverse_diffusion', 'ald', Y.cuda(), N=N, rsp=rsp, taylor_expansion=2,
                corrector_steps=corrector_steps, snr=snr, sampler_type='pc')
            sample, _ = sampler()

                       
                       
        sample = sample.squeeze()
        x_hat = model.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * normfac

        x_hat = x_hat.squeeze().cpu().numpy()
        x = x.squeeze().cpu().numpy()
        y = y.squeeze().cpu().numpy()

        # Both PESQ and DNSMOS need 16khz so we resample here once if necesary
        x_16k = librosa.resample(x, orig_sr=sr, target_sr=16000) if sr != 16000 else x
        x_hat_16k = librosa.resample(x_hat, orig_sr=fs, target_sr=16000) if sr != 16000 else x_hat

        basic_metrics["si_sdr"][i] = si_sdr(x, x_hat)
        try:
            basic_metrics["pesq"][i] = pesq(sr, x_16k, x_hat_16k, 'wb')
        except:
            basic_metrics["pesq"][i] = 0
        #basic_metrics["pesq"][i] = pesq(sr, x_16k, x_hat_16k, 'wb') 
        basic_metrics["estoi"][i] = stoi(x, x_hat, fs, extended=True)


        if i < num_eval_files//2:
            if spec:
                spec_list["y"].append(model._stft(torch.from_numpy(y)))
                spec_list["x_hat"].append(model._stft(torch.from_numpy(x_hat)))
                spec_list["x"].append(model._stft(torch.from_numpy(x)))
                spec_list["fn"].append(fn)
            if audio:
                audio_list["y"].append(y)
                audio_list["x_hat"].append(x_hat)
                audio_list["x"].append(x)
                audio_list["fn"].append(fn)
        i = i+1

        
    return basic_metrics, spec_list, audio_list


