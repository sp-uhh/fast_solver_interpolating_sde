import time
from math import ceil
import warnings
import numpy as np
import torch
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage
import torch.nn.functional as F
from sgmse import sampling
from sgmse.sdes import SDERegistry
from sgmse.backbones import BackboneRegistry
from sgmse.util.inference import evaluate_model
from sgmse.util.other import pad_spec
import numpy as np
import wandb
from sgmse.util.graphics import visualize_example
from utils import MultiResolutionSTFTLoss



class ScoreModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--N_inf", type=int, default=15, help="N reverse steps for inference during training. This is only used for evaluation during training, not for the final evaluation.")
        parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate (1e-4 by default)")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum time (3e-2 by default)")
        parser.add_argument("--num_eval_files", type=int, default=20, help="Number of files for speech enhancement performance evaluation during training. Pass 0 to turn off (no checkpoints based on evaluation metrics will be generated).")
        parser.add_argument("--loss_type", type=str, default="dsm", help="The type of loss function to use.")
        parser.add_argument("--loss_abs_exponent", type=float, default= 0.5,  help="magnitude transformation in the loss term")
        parser.add_argument("--output_scale", type=str, default= 'sigma',  help="backbone model scale in the last output layer")
        parser.add_argument("--timestep_type_inf", type=str, default= 'default',  help="default means to take linear spaced diffusion time steps.")
        parser.add_argument("--audiologs_every_epoch", type=int, help="log audios in wandb every nth epoch")
        parser.add_argument("--speclogs_every_epoch", type=int, help="log specs in wandb every nth epoch")
        parser.add_argument("--loss_weight_type", type=str, default= 'one',  help="diffusion time weightning for the loss")
        parser.add_argument("--sampler_type", type=str, default= 'pc', help="sampler type for inference")
        return parser

    def __init__(
        self, backbone, sde, lr=1e-4, ema_decay=0.999, t_eps=3e-2, loss_abs_exponent=0.5, loss_weight_type='one',
        num_eval_files=20, loss_type='dsm', data_module_cls=None, output_scale='time', N_inf=15, sampler_type='pc',
        timestep_type_inf = 'default', audiologs_every_epoch = 0, speclogs_every_epoch = 0, **kwargs
    ):
        """
        Create a new ScoreModel.

        Args:
            backbone: Backbone DNN that serves as a score-based model.
            sde: The SDE that defines the diffusion process.
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            t_eps: The minimum time to practically run for to avoid issues very close to zero (1e-5 by default).
            loss_type: The type of loss to use (wrt. noise z/std). Options are 'mse' (default), 'mae'
        """
        super().__init__()
        # Initialize Backbone DNN
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        self.dnn = dnn_cls(**kwargs)
        # Initialize SDE
        sde_cls = SDERegistry.get_by_name(sde)
        self.sde = sde_cls(**kwargs)
        # Store hyperparams and save them
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = t_eps
        self.loss_type = loss_type
        self.sampler_type = sampler_type
        self.loss_weight_type = loss_weight_type
        self.num_eval_files = num_eval_files
        self.loss_abs_exponent = loss_abs_exponent
        self.output_scale = output_scale
        self.N_inf = N_inf
        self.save_hyperparameters(ignore=['no_wandb'])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)
        self.timestep_type_inf = timestep_type_inf
        self.audiologs_every_epoch = audiologs_every_epoch
        self.speclogs_every_epoch = speclogs_every_epoch
        self.mrstftloss = MultiResolutionSTFTLoss(factor_sc=.5, factor_mag=.5).to(self.device) #factor para taken from demucs


        


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.parameters())

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                self.ema.store(self.parameters())        # store current params in EMA
                self.ema.copy_to(self.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    
    def _loss(self, score, sigmas, z, gt=None, loss_weight=1):    
        if self.loss_type == 'dsm':
            err = sigmas*score + z 
            losses = torch.square(err.abs())
        elif self.loss_type == 'epsilon':
            pass
            #err = score + loss_weight*z 
            #losses = torch.square(err.abs())
        elif self.loss_type == 'dp':
            if gt is None:
                raise ValueError("gt must be provided for data prediction loss")
            #l2 between score and gt
            losses = torch.square(torch.abs(score - gt))
        else:
            raise ValueError(f'Loss type {self.loss_type} not recognized.')

        # taken from reduce_op function: sum over channels and position and mean over batch dim
        # presumably only important for absolute loss number, not for gradients
        loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        return loss

    def _step(self, batch, batch_idx):
        x, y = batch
        rdm = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps
        t = torch.min(rdm, torch.tensor(self.sde.T))
        mean, std = self.sde.marginal_prob(x, t, y)
        z = torch.randn_like(x)  #
        sigmas = std[:, None, None, None]
        perturbed_data = mean + sigmas * z
        scale = self.get_outputscale(t)
        score = self(perturbed_data, t, y, scale)
        loss_weight = self._get_loss_weight(self.loss_weight_type, t, sigmas)
        loss = self._loss(score, sigmas, z, gt=x, environmental=(y-x), loss_weight=loss_weight)
        return loss

    def _get_loss_weight(self, loss_weight_type, t=None, sigmas=None):
        if loss_weight_type == 'one':
            loss_weight = 1.0
        elif loss_weight_type == 'sigma':
            assert sigmas is not None, "sigmas must be provided for var loss weight"
            loss_weight = sigmas
        else:
            raise ValueError('loss weight type not implemented!')
        return loss_weight
    
    
    def get_outputscale(self, t=None):
        if self.output_scale=='time':
            scale = t[:,None, None, None]
        elif self.output_scale=='sigma':
            std = self.sde._std(t)
            scale  = std[:, None, None, None]
        else:
            scale = None
        return scale
    
    
    
    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True)
        
        if self.num_eval_files != 0 and batch_idx==0:
            basic_metrics, spec, audio = evaluate_model(self, self.num_eval_files)
            for key, value in basic_metrics.items():
                self.log(key, np.mean(value), on_step=False, on_epoch=True)

            if self.audiologs_every_epoch > 0:
                if self.current_epoch % self.audiologs_every_epoch == 0:
                    self._log_audio(audio)
            if self.speclogs_every_epoch > 0:
                if self.current_epoch % self.speclogs_every_epoch == 0:
                    self._log_spec(spec)
        return loss


    def _log_audio(self, audio):
        if audio is not None:
            sr = self.data_module.fs
            for idx, (y, x_hat, x) in enumerate(zip(audio["y"], audio["x_hat"], audio["x"])):
                if type(self.logger).__name__ == "TensorBoardLogger":
                    if self.current_epoch == 0:
                        self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Mix/{idx}", (y * .9/np.max(np.abs(y)))[..., np.newaxis], sample_rate=sr, global_step=None)
                        self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Clean/{idx}", (x * .9/np.max(x))[..., np.newaxis], sample_rate=sr, global_step=None)
                    self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Estimate/{idx}", (x_hat * .9/np.max(np.abs(x_hat)))[..., np.newaxis], sample_rate=sr, global_step=None)
                elif type(self.logger).__name__ == "WandbLogger":
                    if self.current_epoch == 0:
                        self.logger.experiment.log({ f"Audio/Epoch={self.current_epoch}/{idx}": wandb.Audio((y * .9/np.max(np.abs(y)))[..., np.newaxis], caption="Mixture", sample_rate=sr) })
                        self.logger.experiment.log({ f"Audio/Epoch={self.current_epoch}/{idx}": wandb.Audio((x * .9/np.max(np.abs(x)))[..., np.newaxis], caption="Clean", sample_rate=sr) })
                    self.logger.experiment.log({ f"Audio/Epoch={self.current_epoch}/{idx}": wandb.Audio((x_hat * .9/np.max(np.abs(x_hat)))[..., np.newaxis], caption="Estimate", sample_rate=sr) })


    def _log_spec(self, spec):
        if spec is not None:
            figures = []
            for idx, (y_stft, x_hat_stft, x_stft) in enumerate(zip(spec["y"], spec["x_hat"], spec["x"])):
                figures.append(
                    visualize_example(
                    torch.abs(y_stft), 
                    torch.abs(x_hat_stft), 
                    torch.abs(x_stft), 
                    sample_rate=self.data_module.fs,
                    hop_len=self.data_module.hop_length,
                    return_fig=True)
                    )
            if type(self.logger).__name__ == "TensorBoardLogger":
                self.logger.experiment.add_figure(f"Epoch={self.current_epoch}", figures, close=True, global_step=None)
            elif type(self.logger).__name__ == "WandbLogger":
                self.logger.experiment.log({f"Spec/Epoch={self.current_epoch}": [wandb.Image(fig, caption=f"Sample {i}") for (i, fig) in enumerate(figures)]})




    def forward(self, x, t, y, divide_scale):
        out = self.dnn(x, t, y, divide_scale)
        return out
    
    def forward_inference(self, x, t, y, divide_scale, return_what = 'score'):
        out = self.dnn(x, t, y, divide_scale)
        if return_what == 'score':
            if self.loss_type == 'dsm':
                return out
            elif self.loss_type == 'dp':
                sigma = self.sde._std(t)[:, None, None, None]
                mean_t  = self.sde._mean(out,t,y)
                score = -(x - mean_t)/sigma**2
                return score
        else:
            raise ValueError('To be implemented other return variables')
            
        

    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def get_sampler(self, predictor_name, corrector_name, y, kappa=1.0,
                       Y_prior=None, N=None, sampler_type='pc', adaptive_ode_method='RK45',
                       rsp=1.0, **kwargs):
        sde = self.sde.copy()
        sde.N = N

        #kwargs = {"teps": self.t_eps, **kwargs}
                
        if sampler_type == 'pc':
            return sampling.get_pc_sampler(predictor_name, corrector_name, sde, self,
                y, rsp=rsp, Y_prior=Y_prior, teps=self.t_eps, **kwargs)
        elif sampler_type == 'adaptive':
            return sampling.get_ode_sampler(
                    sde, self, y, Y_prior=Y_prior, inverse_scaler=None, 
                    denoise=True, teps=self.t_eps, 
                    method=adaptive_ode_method, device='cuda', **kwargs
                )
        elif sampler_type in ['ISDE_2steps']:
            return sampling.get_ISDE_sampler(
                sde=sde, score_fn=self, Y=y, 
                Y_prior=Y_prior, rsp=rsp, teps=self.t_eps,
                kappa=kappa, sampler_type=sampler_type,
                **kwargs)
            
        elif sampler_type in ['rk23', 'rk45', 'midpoint']:
            return sampling.get_rk_sampler(
                sde=sde, score_fn=self, Y=y, 
                Y_prior=Y_prior, rsp=rsp, teps=self.t_eps,
                kappa=kappa, sampler_type = sampler_type,
                **kwargs)
        raise ValueError(f'Sampler type {sampler_type} not recognized.')
        



    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)




    def enhance(self, y, sampler_type="pc", predictor="reverse_diffusion", kappa=1,
        corrector="ald", N=30, corrector_steps=1, snr=0.5, timeit=False, adaptive_ode_method='RK45',
        **kwargs
    ):
        """
        One-call speech enhancement of noisy speech `y`, for convenience.
        """
        start = time.time()
        T_orig = y.size(1) 
        norm_factor = y.abs().max().item()
        y = y / norm_factor
        
        Y = torch.unsqueeze(self._forward_transform(self._stft(y.cuda())), 0)
        Y = pad_spec(Y)
        
    
        sampler = self.get_sampler(predictor, corrector, Y.cuda(), N=N, rsp = self.sde.T, kappa=kappa,
            corrector_steps=corrector_steps, snr=snr, intermediate=False,  sampler_type=sampler_type, adaptive_ode_method=adaptive_ode_method,
            **kwargs)

        sample, nfe = sampler()
        
        
        sample = sample.squeeze()
        
        x_hat = self.to_audio(sample, T_orig)
        x_hat = x_hat * norm_factor
        x_hat = x_hat.squeeze().cpu().numpy()
        end = time.time()
        if timeit:
            proc_time = (end-start)
            return x_hat, nfe, proc_time
        else:
            return x_hat
        
