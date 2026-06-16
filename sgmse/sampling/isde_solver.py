
from scipy.integrate import quad, trapezoid
import abc
from scipy.special import factorial
import torch
import numpy as np

from sgmse.util.registry import Registry

ISDESolverRegistry = Registry("ISDESolverRegistry")

class ISDE_solver(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, model, kappa=1):
        super().__init__()
        self.sde = sde
        self.model = model
        self.kappa = kappa


    @abc.abstractmethod
    def update_fn(self, x, t, *args):
        """One update of the predictor.

        Args:
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.
            *args: Possibly additional arguments, in particular `y` for OU processes
        """
        pass

    def debug_update_fn(self, x, t, *args):
        raise NotImplementedError(f"Debug update function not implemented for predictor {self}.")


@ISDESolverRegistry.register('dpm_2s')
class ISDE_2steps(ISDE_solver):
    def __init__(self, sde, model, kappa):
        super().__init__(sde, model, kappa)

    
    def compute_linear_term(self,t, s, xs, Y):
        assert t < s
        psi = (1-self.sde.kt(t))/(1-self.sde.kt(s))
        linear_term = xs*psi + (1-psi)*Y
        return linear_term

    
    def update_fn(self, x_upper, vec_upper, Y, upper_int_limit = None,
                  lower_int_limit=None, loss='dsm', last_step=False):
        
        kappa = self.kappa
        stepsize = upper_int_limit - lower_int_limit
        gamma=0.5  #0.5 correspond to midpoint
        midpoint = upper_int_limit - gamma*stepsize 
        random_part = 0
        num_eval_ito_integral_2 = 0
        
        # first evaluation from upper_point = s to midpoint, needed for second taylor expansion
        score_upper = self.model.forward_inference(x_upper, vec_upper, Y, self.model.get_outputscale(vec_upper), return_what = 'score')

        
        if hasattr(self.sde, 'exact_fouve_integral') and callable(getattr(self.sde, 'exact_fouve_integral')):
            def_integral = self.sde.exact_fouve_integral(0, midpoint, upper_int_limit)
            num_eval_integrand = 0  # or some default value
        else:
            def_integral, num_eval_integrand = _integral(self._integrand, midpoint, upper_int_limit, 
                                                  int_method='quad', n=0, loss=loss)
        
        
        
        lin_term = self.compute_linear_term(midpoint, upper_int_limit, x_upper, Y)
        x_midpoint = lin_term + def_integral*score_upper*(1-self.sde.kt(midpoint))
        
        
        #second evaluation
        vec_mid = torch.ones(Y.shape[0], device=Y.device) * midpoint
        score_mid = self.model.forward_inference(x_midpoint, vec_mid, Y, self.model.get_outputscale(vec_mid), return_what = 'score')
        
        #first taylor term
        if hasattr(self.sde, 'exact_weight_integral') and callable(getattr(self.sde, 'exact_weight_integral')):
            def_integral_0 = self.sde.exact_weight_integral(0, lower_int_limit, upper_int_limit)
            num_eval_integrand_0 = 0  # or some default value
        else:
            def_integral_0, num_eval_integrand_0 = _integral(self._integrand, lower_int_limit, upper_int_limit, int_method='quad', n=0, loss=loss)
        
        taylor_0 = def_integral_0*score_upper*(1-self.sde.kt(lower_int_limit))
        
        #second taylor term
        if hasattr(self.sde, 'exact_weight_integral') and callable(getattr(self.sde, 'exact_weight_integral')):
            def_integral_1 = self.sde.exact_weight_integral(1, lower_int_limit, upper_int_limit)
            # Optionally, you can also capture num_eval_integrand if needed
            num_eval_integrand_1 = 0  # or some default value
        else:
            def_integral_1, num_eval_integrand_1 = _integral(self._integrand, lower_int_limit, upper_int_limit, int_method='quad', n=1, loss=loss)
        
        first_derivative =(score_upper - score_mid)/(gamma*stepsize)
        taylor_1 = def_integral_1*first_derivative*(1-self.sde.kt(lower_int_limit))
        lin_term = self.compute_linear_term(lower_int_limit, upper_int_limit, x_upper, Y)

        if kappa != 0 and not last_step:
            if hasattr(self.sde, 'ito_integral') and callable(getattr(self.sde, 'ito_integral')):
                ito_integral_2 = self.sde.ito_integral(lower_int_limit, upper_int_limit)
                num_eval_ito_integral_2 = 0
            else:
                ito_integral_2, num_eval_ito_integral_2 = _integral(self._ito_integral_squared, lower_int_limit, upper_int_limit, int_method='quad') 
                
            z = torch.randn_like(x_upper)
            random_part = kappa*(1-self.sde.kt(lower_int_limit))*np.sqrt(ito_integral_2)*z
        if last_step:
            random_part = 0
            kappa = 0
            
        x_lower = lin_term + (taylor_0 + taylor_1)*((1+kappa**2)) + random_part
        
        return x_lower, num_eval_integrand_1 + num_eval_integrand_0 + num_eval_integrand + num_eval_ito_integral_2 
     

    def _integrand(self, tau, tau_0, n, loss='dsm'):
        #tau_0 should be the lower integral limit
        #tau is the integration variable
        #n is the order of the taylor expansion
        A = np.power(tau - tau_0, n)/factorial(n,exact=True)
        if loss == 'dsm':
            out = 0.5*A * (self.sde.diffusion_np(tau)**2)/((1-self.sde.kt_np(tau)))
        elif loss == 'epsilon':            
            pass
        elif loss == 'dp':            
            pass
        return out
    
    def _ito_integral_squared(self, tau):
        return (self.sde.diffusion_np(tau)/(1-self.sde.kt_np(tau)))**2

    def update_fn_analyze(self, x, t, *args):
        raise NotImplementedError("update_fn_analyze() has not been implemented yet for the ReverseDiffusionPredictor")


    

def _integral(integrand, lower, upper, int_method='quad', **kwargs):
    lower_float = float(lower)
    upper_float = float(upper)
    if kwargs:
        k, v=zip(*kwargs.items())
    if int_method == 'quad':
        if kwargs:
            out = quad(integrand, lower_float, upper_float, args=(upper_float, *v), full_output=1)
        else:
            out = quad(integrand, lower_float, upper_float,  full_output=1)
        return out[0], out[2]['neval']
    elif int_method == 'trapezoid':
        x_array = np.linspace(lower_float, upper_float, num=1000)
        if kwargs:
            y = integrand(x_array, lower_float, *v)
        else:
            y = integrand(x_array, lower_float)
        integral_out = trapezoid(y, x_array)
        return integral_out, 1000
    else:
        raise ValueError(f"Integration method {int_method} not recognized. Use 'quad' or 'trapezoid'.")
    
    