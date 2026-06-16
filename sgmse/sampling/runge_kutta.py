
import abc
from scipy.special import factorial
import torch
import numpy as np

from sgmse.util.registry import Registry

RKRegistry = Registry("RK")

class RK_solver(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, model, a,b,c, kappa=1):
        super().__init__()
        self.sde = sde
        self.model = model
        self.kappa = kappa
        self.a = a.to(device='cuda')
        self.b = b.to(device='cuda')
        self.c = c.to(device='cuda')


    @abc.abstractmethod
    def update_fn(self, x, t, *args):
        """One update of the predictor.

        Args:
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.
            *args: Possibly additional arguments, in particular `y` for OU processes

        Returns:
            x: A PyTorch tensor of the next state.
            x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass

    def debug_update_fn(self, x, t, *args):
        raise NotImplementedError(f"Debug update function not implemented for predictor {self}.")


@RKRegistry.register('rk')
class RK(RK_solver):
    def __init__(self, sde, model, a,b,c, kappa):
        super().__init__(sde, model, a,b,c, kappa)

    def update_fn(self, x_upper, Y, 
                  upper_int_limit = None, 
                  lower_int_limit=None):
        
        dt = -(upper_int_limit - lower_int_limit)
        q = self.a.shape[0]
        k_list = []
        for i in range(q):
            if i == 0:
                stage_input = x_upper
            else:
                k_prev = torch.stack(k_list[:i], dim=0)  # (q, B, C, F, T)
                weighted_sum = torch.sum(self.a[i, :i][:, None, None, None, None] * k_prev, dim=0)
                stage_input = x_upper + dt * weighted_sum
            t_i = upper_int_limit + self.c[i:i+1] * dt
            k_i = self.v_theta(stage_input, t_i, Y)
            k_list.append(k_i)
        k_all = torch.stack(k_list, dim=0)  # (q, B, C, F, T)
        k_total = torch.sum(self.b[:, None, None, None, None] * k_all, dim=0)
        out = x_upper + dt * k_total
        return out

    def v_theta(self, stage_input, t_i, Y):
        #vectorized version of t_i as tensor
        t_i_vector = torch.ones(Y.shape[0], device=Y.device) * t_i
        #[f(x_,t ,t) - g(t)**2 * score_model(x_, t)/2]dt
        score = self.model.forward_inference(stage_input, t_i_vector, Y, self.model.get_outputscale(t_i_vector), return_what = 'score')
        f, g = self.sde.sde(stage_input, t_i, Y)
        out = f - 0.5 * (g**2) * score
        return out
    