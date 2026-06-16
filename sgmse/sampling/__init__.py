# Adapted from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sampling.py
"""Various sampling methods."""
from scipy import integrate
import torch

from sgmse.sampling.butcher_tables import get_butcher_tableau
from sgmse.sampling.runge_kutta import RK
from .isde_solver import ISDE_2steps
from .predictors import Predictor, PredictorRegistry, ReverseDiffusionPredictor
from .correctors import Corrector, CorrectorRegistry
import numpy as np
import matplotlib.pyplot as plt


__all__ = [
    'PredictorRegistry', 'CorrectorRegistry', 'Predictor', 'Corrector',
    'get_sampler'
]


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))


def get_pc_sampler(
    predictor_name, corrector_name, sde, score_fn, Y, Y_prior=None, rsp=1.0,
    denoise=True, teps=3e-2, snr=0.1, corrector_steps=1, probability_flow: bool = False, **kwargs
):
    """Create a Predictor-Corrector (PC) sampler.

    Args:
        predictor_name: The name of a registered `sampling.Predictor`.
        corrector_name: The name of a registered `sampling.Corrector`.
        sde: An `sdes.SDE` object representing the forward SDE.
        score_fn: A function (typically learned model) that predicts the score.
        y: A `torch.Tensor`, representing the (non-white-)noisy starting point(s) to condition the prior on.
        denoise: If `True`, add one-step denoising to the final samples.
        eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
        snr: The SNR to use for the corrector. 0.1 by default, and ignored for `NoneCorrector`.
        N: The number of reverse sampling steps. If `None`, uses the SDE's `N` property by default.

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """
    predictor_cls = PredictorRegistry.get_by_name(predictor_name)
    corrector_cls = CorrectorRegistry.get_by_name(corrector_name)
    predictor = predictor_cls(sde, score_fn, probability_flow=probability_flow)
    corrector = corrector_cls(sde, score_fn, snr=snr, n_steps=corrector_steps)
    model = score_fn

    def pc_sampler():
        """The PC sampler function."""
        with torch.no_grad():
            

            Y_prior = Y
            xt, _ = sde.prior_sampling(Y_prior.shape, Y_prior)
            timesteps = timesteps_space(rsp, sde.N,teps, Y.device, type=model.timestep_type_inf)
            xt = xt.to(Y_prior.device)
            
            
            for i in range(len(timesteps)):
                t = timesteps[i]
                if i != len(timesteps) - 1:
                    stepsize = t - timesteps[i+1]
                else:
                    stepsize = timesteps[-1]

                vec_t = torch.ones(Y.shape[0], device=Y.device) * t

                xt, xt_mean = corrector.update_fn(xt, vec_t, Y)
                xt, xt_mean = predictor.update_fn(xt, vec_t, Y, stepsize)
            x_result = xt_mean if denoise else xt
            ns = len(timesteps) * (corrector.n_steps + 1)
            return x_result, ns

    return pc_sampler




def timesteps_space(sdeT, sdeN,  eps, device, type='linear'):
    timesteps = torch.linspace(sdeT, eps, sdeN, device=device)
    if type == 'default':
        return timesteps
    else:
        raise ValueError(f'Unknown timestep spacing type {type}.')


def get_ISDE_sampler(
    sde, score_fn, Y, teps=3e-2, rsp=1.0, kappa=2, sampler_type='2steps', **kwargs):
    """DPM-Solver sampler.

    Args:
        sde: An `sdes.SDE` object representing the forward SDE.
        score_fn: A function (typically learned model) that predicts the score.
        y: A `torch.Tensor`, representing the (non-white-)noisy starting point(s) to condition the prior on.
        timestep_type: Type of timestep spacing.

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """


    if sampler_type == 'ISDE_2steps':
        sampler_obj = ISDE_2steps(sde, score_fn, kappa=kappa)
    else:
        raise ValueError(f'Unknown sampler type {sampler_type} for DPM solver.')
    def ISDE_sampler():
        """ISDE sampler: "A Fast Solver for Interpolating Stochastic Differential Equation Diffusion Models for Speech Restoration" by Lay&Gerkmann """
        with torch.no_grad():
            

            Y_prior = Y            
            xs, _ = sde.prior_sampling(Y_prior.shape, Y_prior)
            xs = xs.to(Y_prior.device)

            timesteps = timesteps_space(rsp, sde.N, teps, Y.device, type=score_fn.timestep_type_inf)
            integrand_nfes_sum = 0
            last_step = False
            for i in range(len(timesteps)):
                if i < len(timesteps)-1:
                    s = timesteps[i]  #s > t
                    t = timesteps[i+1]
                else:
                    #last stop to t=0
                    s = timesteps[i]
                    t = timesteps[i]*0
                    last_step = True
                
                vec_s = torch.ones(Y.shape[0], device=Y.device) * s
                xs, integrand_nfes = sampler_obj.update_fn(xs, vec_s, Y, 
                                            upper_int_limit = s,
                                            lower_int_limit= t, 
                                            loss=score_fn.loss_type, last_step=last_step)

                integrand_nfes_sum += integrand_nfes

            nfe = 2*len(timesteps)  #two model evaluations per step
            #print(integrand_nfes_sum) number of evaluation needed to compute \omega_n in eq (29).
            
            x_result = xs
            return x_result, nfe
    return ISDE_sampler




def get_rk_sampler(
    sde, score_fn, Y, teps=3e-2, rsp=1.0, kappa=2, sampler_type='rk', **kwargs):
    """DPM-Solver sampler.

    Args:
        sde: An `sdes.SDE` object representing the forward SDE.
        score_fn: A function (typically learned model) that predicts the score.
        y: A `torch.Tensor`, representing the (non-white-)noisy starting point(s) to condition the prior on.
        timestep_type: Type of timestep spacing.

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """

    
    a,b,c = get_butcher_tableau(sampler_type)
    sampler_obj = RK(sde, score_fn, a,b,c,kappa=kappa)
    def rk_sampler():
        with torch.no_grad():
            Y_prior = Y            
            xs, _ = sde.prior_sampling(Y_prior.shape, Y_prior)
            xs = xs.to(Y_prior.device)

            timesteps = timesteps_space(rsp, sde.N, teps, Y.device, type=score_fn.timestep_type_inf)
            for i in range(len(timesteps)):
                if i < len(timesteps)-1:
                    s = timesteps[i]  #s > t
                    t = timesteps[i+1]
                else:
                    #last stop to t=0
                    s = timesteps[i]
                    t = timesteps[i]*0

                xs = sampler_obj.update_fn(xs, Y, 
                                            upper_int_limit = s,
                                            lower_int_limit= t)



            nfe = 2*len(timesteps)  #two model evaluations per step
            x_result = xs
            return x_result, nfe

    return rk_sampler







def get_ode_sampler(
    sde, score_fn, y, inverse_scaler=None, 
    denoise=True, rtol=1e-5, atol=1e-5, 
    method='RK45', device='cuda', teps=1e-3, **kwargs
):
    """Probability flow ODE sampler with the black-box ODE solver.

    Args:
        sde: An `sdes.SDE` object representing the forward SDE.
        score_fn: A function (typically learned model) that predicts the score.
        y: A `torch.Tensor`, representing the (non-white-)noisy starting point(s) to condition the prior on.
        inverse_scaler: The inverse data normalizer.
        denoise: If `True`, add one-step denoising to final samples.
        rtol: A `float` number. The relative tolerance level of the ODE solver.
        atol: A `float` number. The absolute tolerance level of the ODE solver.
        method: A `str`. The algorithm used for the black-box ODE solver.
            See the documentation of `scipy.integrate.solve_ivp`.
        eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
        device: PyTorch device.

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """
    predictor = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
    rsde = sde.reverse(score_fn, probability_flow=True)

    def denoise_update_fn(x):
        vec_eps = torch.ones(x.shape[0], device=x.device) * teps
        _, x = predictor.update_fn(x, vec_eps, y, 0.03)
        return x

    def drift_fn(x, t, y):
        """Get the drift function of the reverse-time SDE."""
        return rsde.sde(x, t, y)[0]

    def ode_sampler(z=None, **kwargs):
        """The probability flow ODE sampler with black-box ODE solver.

        Args:
            model: A score model.
            z: If present, generate samples from latent code `z`.
        Returns:
            samples, number of function evaluations.
        """
        with torch.no_grad():
            # If not represent, sample the latent code from the prior distibution of the SDE.



            Y_prior = y
            xt, _ = sde.prior_sampling(Y_prior.shape, Y_prior)
            x = xt.to(Y_prior.device)
            
            timesteps = timesteps_space(sde.T, sde.N, teps, x.device, type=score_fn.timestep_type_inf)
            t_np = timesteps.cpu().numpy()
            

            def ode_func(t, x):
                x = from_flattened_numpy(x, y.shape).to(device).type(torch.complex64)
                vec_t = torch.ones(y.shape[0], device=x.device) * t
                drift = drift_fn(x, vec_t, y)
                return to_flattened_numpy(drift)

            # Black-box ODE solver for the probability flow ODE
            solution = integrate.solve_ivp(
                ode_func, (sde.T, teps), to_flattened_numpy(x), t_eval = t_np,
                rtol=rtol, atol=atol, method=method, dense_output=True, **kwargs
            )

            nfe = solution.nfev
            x = torch.tensor(solution.y[:, -1]).reshape(y.shape).to(device).type(torch.complex64)

            # Denoising is equivalent to running one predictor step without adding noise
            if denoise:
                x = denoise_update_fn(x)

            if inverse_scaler is not None:
                x = inverse_scaler(x)
            return x, nfe

    return ode_sampler
