# Copyright 2022 UC Berkeley Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# DISCLAIMER: This file is strongly influenced by https://github.com/ermongroup/ddim

from logging import Logger
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch

from ..configuration_utils import ConfigMixin, FrozenDict, register_to_config
from ..utils import _COMPATIBLE_STABLE_DIFFUSION_SCHEDULERS, BaseOutput, deprecate,logger
from .scheduling_utils import SchedulerMixin

#logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@dataclass
class SGHMCSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's step function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample (x_{0}) based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """
    momentum: torch.FloatTensor
    prev_sample: torch.FloatTensor
    pred_original_sample: Optional[torch.FloatTensor] = None


def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class SGHMCScheduler(SchedulerMixin, ConfigMixin):
    """
    Stochastic Gradient Hamilton Monte Carlo with Langevin dynamics
    http://proceedings.mlr.press/v32/cheni14.pdf
    
    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`SchedulerMixin`] provides general loading and saving functionality via the [`SchedulerMixin.save_pretrained`] and
    [`~SchedulerMixin.from_pretrained`] functions.

    For more details, see the original paper: https://arxiv.org/abs/2006.11239

    Args:
        num_train_timesteps (`int`): number of diffusion steps used to train the model.
        beta_start (`float`): the starting `beta` value of inference.
        beta_end (`float`): the final `beta` value.
        beta_schedule (`str`):
            the beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
        trained_betas (`np.ndarray`, optional):
            option to pass an array of betas directly to the constructor to bypass `beta_start`, `beta_end` etc.
        variance_type (`str`):
            options to clip the variance used when adding noise to the denoised sample. Choose from `fixed_small`,
            `fixed_small_log`, `fixed_large`, `fixed_large_log`, `learned` or `learned_range`.
        clip_sample (`bool`, default `True`):
            option to clip predicted sample between -1 and 1 for numerical stability.
        prediction_type (`str`, default `epsilon`, optional):
            prediction type of the scheduler function, one of `epsilon` (predicting the noise of the diffusion
            process), `sample` (directly predicting the noisy sample`) or `v_prediction` (see section 2.4
            https://imagen.research.google/video/paper.pdf)
    """
    order = 2
    _compatibles = _COMPATIBLE_STABLE_DIFFUSION_SCHEDULERS.copy()
    _deprecated_kwargs = ["predict_epsilon"]

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[np.ndarray] = None,
        variance_type: str = "fixed_small",
        clip_sample: bool = True,
        prediction_type: str = "epsilon",
        sampler_type:str ="sde_vp",
        **kwargs,
    ):
        message = (
            "Please make sure to instantiate your scheduler with `prediction_type` instead. E.g. `scheduler ="
            " DDPMScheduler.from_pretrained(<model_id>, prediction_type='epsilon')`."
        )
        predict_epsilon = deprecate("predict_epsilon", "0.11.0", message, take_from=kwargs)
        if predict_epsilon is not None:
            self.register_to_config(prediction_type="epsilon" if predict_epsilon else "sample")

        if trained_betas is not None:
            self.betas = torch.from_numpy(trained_betas)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = (
                torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
            )
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        elif beta_schedule == "sigmoid":
            # GeoDiff sigmoid schedule
            betas = torch.linspace(-6, 6, num_train_timesteps)
            self.betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # setable values
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())

        self.variance_type = variance_type

        if(sampler_type=="sde_ve"):
            sigma_min: float = 0.01,
            sigma_max: float = 1348.0,
            sampling_eps: float = 1e-5,
            self.set_sigmas(num_train_timesteps, sigma_min, sigma_max, sampling_eps)
        
        if(sampler_type=="euler_discrete"):
            self.is_scale_input_called = False

    def set_sigmas(
        self, num_inference_steps: int, sigma_min: float = None, sigma_max: float = None, sampling_eps: float = None
    ):
        """
        Sets the noise scales used for the diffusion chain. Supporting function to be run before inference.

        The sigmas control the weight of the `drift` and `diffusion` components of sample update.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
            sigma_min (`float`, optional):
                initial noise scale value (overrides value given at Scheduler instantiation).
            sigma_max (`float`, optional): final noise scale value (overrides value given at Scheduler instantiation).
            sampling_eps (`float`, optional): final timestep value (overrides value given at Scheduler instantiation).

        """
        sigma_min = sigma_min if sigma_min is not None else self.config.sigma_min
        sigma_max = sigma_max if sigma_max is not None else self.config.sigma_max
        sampling_eps = sampling_eps if sampling_eps is not None else self.config.sampling_eps
        if self.timesteps is None:
            self.set_timesteps(num_inference_steps, sampling_eps)

        self.sigmas = sigma_min * (sigma_max / sigma_min) ** (self.timesteps / sampling_eps)
        self.discrete_sigmas = torch.exp(torch.linspace(math.log(sigma_min), math.log(sigma_max), num_inference_steps))
        self.sigmas = torch.tensor([sigma_min * (sigma_max / sigma_min) ** t for t in self.timesteps])


    def scale_model_input(self, sample: torch.FloatTensor, timestep: Optional[int] = None) -> torch.FloatTensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.FloatTensor`): input sample
            timestep (`int`, optional): current timestep

        Returns:
            `torch.FloatTensor`: scaled input sample
        """
        return sample

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """
        num_inference_steps = min(self.config.num_train_timesteps, num_inference_steps)
        self.num_inference_steps = num_inference_steps
        timesteps = np.arange(
            0, self.config.num_train_timesteps, self.config.num_train_timesteps // self.num_inference_steps
        )[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps).to(device)

        if(self.sampler_type=="euler_discrete"):
            self.num_inference_steps = num_inference_steps
            timesteps = np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps, dtype=float)[::-1].copy()
            sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
            sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
            sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
            self.sigmas = torch.from_numpy(sigmas).to(device=device)
            if str(device).startswith("mps"):
                # mps does not support float64
                self.timesteps = torch.from_numpy(timesteps).to(device, dtype=torch.float32)
            else:
                self.timesteps = torch.from_numpy(timesteps).to(device=device)


    def scale_model_input(
        self, sample: torch.FloatTensor, timestep: Union[float, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """
        Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the Euler algorithm.

        Args:
            sample (`torch.FloatTensor`): input sample
            timestep (`float` or `torch.FloatTensor`): the current timestep in the diffusion chain

        Returns:
            `torch.FloatTensor`: scaled input sample
        """
        if(self.sampler_type=="euler_discretre"):
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            step_index = (self.timesteps == timestep).nonzero().item()
            sigma = self.sigmas[step_index]
            sample = sample / ((sigma**2 + 1) ** 0.5)
            self.is_scale_input_called = True

        return sample

    def _get_variance(self, t, predicted_variance=None, variance_type=None):
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[t - 1] if t > 0 else self.one

        # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * self.betas[t]

        if variance_type is None:
            variance_type = self.config.variance_type

        # hacks - were probably added for training stability
        if variance_type == "fixed_small":
            variance = torch.clamp(variance, min=1e-20)
        # for rl-diffuser https://arxiv.org/abs/2205.09991
        elif variance_type == "fixed_small_log":
            variance = torch.log(torch.clamp(variance, min=1e-20))
            variance = torch.exp(0.5 * variance)
        elif variance_type == "fixed_large":
            variance = self.betas[t]
        elif variance_type == "fixed_large_log":
            # Glide max_log
            variance = torch.log(self.betas[t])
        elif variance_type == "learned":
            return predicted_variance
        elif variance_type == "learned_range":
            min_log = variance
            max_log = self.betas[t]
            frac = (predicted_variance + 1) / 2
            variance = frac * max_log + (1 - frac) * min_log

        return variance

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        p: torch.FloatTensor=None,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator=None,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[SGHMCSchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            generator: random number generator.
            return_dict (`bool`): option for returning tuple rather than DDPMSchedulerOutput class

        Returns:
            [`~schedulers.scheduling_utils.DDPMSchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.DDPMSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.

        """
        #### DDPM ####
        if(self.sampler_type == "ddpm" or self.sampler_type == "DDPM"):
            t = timestep
            dt=1.
            dump_coef=0.2

            if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
                model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
            else:
                predicted_variance = None

            # 1. compute alphas, betas
            alpha_prod_t = self.alphas_cumprod[t]
            alpha_prod_t_prev = self.alphas_cumprod[t - 1] if t > 0 else self.one
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev

            # 2. compute predicted original sample from predicted noise also called
            # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
            if self.config.prediction_type == "epsilon":
                pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            elif self.config.prediction_type == "sample":
                pred_original_sample = model_output
            elif self.config.prediction_type == "v_prediction":
                pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                    " `v_prediction`  for the DDPMScheduler."
                )

            # 3. Clip "predicted x_0"
            if self.config.clip_sample:
                pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

            # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
            # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * self.betas[t]) / beta_prod_t
            current_sample_coeff = self.alphas[t] ** (0.5) * beta_prod_t_prev / beta_prod_t

            # 5. Compute predicted previous sample µ_t
            # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

            # 6. Add noise
            variance = 0
            if t > 0:
                device = model_output.device
                if device.type == "mps":
                    # randn does not work reproducibly on mps
                    variance_noise = torch.randn(model_output.shape, dtype=model_output.dtype, generator=generator)
                    variance_noise = variance_noise.to(device)
                else:
                    variance_noise = torch.randn(
                        model_output.shape, generator=generator, device=device, dtype=model_output.dtype
                    )
                if self.variance_type == "fixed_small_log":
                    variance =  self._get_variance(t, predicted_variance=predicted_variance)  * variance_noise
                else:
                    variance = (self._get_variance(t, predicted_variance=predicted_variance) ** 0.5) * variance_noise
            
            pdif=((pred_prev_sample-sample) -p + variance)
            prev_p= p + pdif*dump_coef
            prev_sample= sample+prev_p
            logger.debug("mean %g,var%g"%(pdif.mean(),pdif.var()))
            #            from IPython.core.debugger import Pdb; Pdb().set_trace()
            #prev_sample= pred_prev_sample + variance

        elif(self.sampler_type == "euler" or self.sampler_type == "euler_discrete"):
    ## from euler discrete code
            if (
                isinstance(timestep, int)
                or isinstance(timestep, torch.IntTensor)
                or isinstance(timestep, torch.LongTensor)
            ):
                raise ValueError(
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep.",
                )

#            if not self.is_scale_input_called:
#                print(
#                    "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
#                    "See `StableDiffusionPipeline` for a usage example."
#                )

            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)

            step_index = (self.timesteps == timestep).nonzero().item()
            sigma = self.sigmas[step_index]

            gamma = min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0

            device = model_output.device
            if device.type == "mps":
                # randn does not work reproducibly on mps
                noise = torch.randn(model_output.shape, dtype=model_output.dtype, device="cpu", generator=generator).to(
                    device
                )
            else:
                noise = torch.randn(model_output.shape, dtype=model_output.dtype, device=device, generator=generator).to(
                    device
                )

            eps = noise * s_noise
            sigma_hat = sigma * (gamma + 1)

            if gamma > 0:
                var= eps * (sigma_hat**2 - sigma**2) ** 0.5
            else:            
                var = 0

            neps= eps * var

            # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
            if self.config.prediction_type == "epsilon":
                pred_original_sample = sample - sigma_hat * model_output
            elif self.config.prediction_type == "v_prediction":
                # * c_out + input * c_skip
                pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
                )

            #sample = sample + neps
            # 2. Convert to an ODE derivative

            derivative = (sample - pred_original_sample) / sigma_hat
            dt = self.sigmas[step_index + 1] - sigma_hat

            #momentum
            prev_p= (derivative -var*p )*dt  + neps
            #position (mass=1)
            prev_sample = sample + prev_p*dt

        elif("sde_ve" in self.sampler_type):
    #### from step_pred of sde_ve
            if self.timesteps is None:
                    raise ValueError(
                    "`self.timesteps` is not set, you need to run 'set_timesteps' after creating the scheduler"
                )

            timestep = timestep * torch.ones(
                sample.shape[0], device=sample.device
            )  # torch.repeat_interleave(timestep, sample.shape[0])
            timesteps = (timestep * (len(self.timesteps) - 1)).long()

            # mps requires indices to be in the same device, so we use cpu as is the default with cuda
            timesteps = timesteps.to(self.discrete_sigmas.device)

            sigma = self.discrete_sigmas[timesteps].to(sample.device)
            adjacent_sigma = self.get_adjacent_sigma(timesteps, timestep).to(sample.device)
            drift = torch.zeros_like(sample)
            diffusion = (sigma**2 - adjacent_sigma**2) ** 0.5

            # equation 6 in the paper: the model_output modeled by the network is grad_x log pt(x) 
            # (model_output is score function)
            # also equation 47 shows the analog from SDE models to ancestral sampling methods
            diffusion = diffusion.flatten()
            while len(diffusion.shape) < len(sample.shape):
                diffusion = diffusion.unsqueeze(-1)
            drift = drift - diffusion**2 * model_output

            #  equation 6: sample noise for the diffusion term of
            noise = torch.randn(sample.shape, layout=sample.layout, generator=generator).to(sample.device)

            #prev_sample_mean = sample - drift  # subtract because `dt` is a small negative timestep
            # TODO is the variable diffusion the correct scaling term for the noise?
            
            prev_p=  p - drift - diffusion*diffusion*p + diffusion * noise  # add impact of diffusion field g
            prev_sample = sample+prev_p

        else: 
    #### from step_pred of sde_vp
            if self.timesteps is None:
                raise ValueError(
                    "`self.timesteps` is not set, you need to run 'set_timesteps' after creating the scheduler"
                )

            t=timestep
            pred_original_sample=model_output
            # TODO(Patrick) better comments + non-PyTorch
            # postprocess model score
            log_mean_coeff = (
                -0.25 * t**2 * (self.betas[-1] - self.betas[0]) - 0.5 * t * self.betas[0]
#                -0.25 * t**2 * (self.config.beta_max - self.config.beta_min) - 0.5 * t * self.config.beta_min
            )
            std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
            std = std.flatten()
            while len(std.shape) < len(model_output.shape):
                std = std.unsqueeze(-1)
            score_norm = -model_output / std

            # compute
            dt = -1.0 / len(self.timesteps)
            beta_t = self.betas[0] + t * (self.betas[-1] - self.betas[0])
            #beta_t = self.config.beta_min + t * (self.config.beta_max - self.config.beta_min)
            beta_t = beta_t.flatten()
            while len(beta_t.shape) < len(sample.shape):
                beta_t = beta_t.unsqueeze(-1)
            drift = -0.5 * beta_t * sample

            diffusion = torch.sqrt(beta_t)
            drift = drift - diffusion**2 * score_norm
 
            # add noise
            device = model_output.device
            if device.type == "mps":
                # randn does not work reproducibly on mps
                noise = torch.randn(sample.shape, layout=sample.layout, generator=generator).to(device)
            else:
                noise = torch.randn(sample.shape, layout=sample.layout, device=device, generator=generator)
            #prev_p=  p + (drift * dt) - p*dt + diffusion * math.sqrt(-dt) * noise
            prev_p=  p + (drift * dt) - diffusion*diffusion*p*dt + diffusion * math.sqrt(-dt) * noise
            #prev_sample =sample +prev_p*dt 
            prev_sample =sample +drift*dt + diffusion * math.sqrt(-dt) * noise

            #x_mean = x + drift * dt
            #x = x_mean + diffusion * math.sqrt(-dt) * noise
        if not return_dict:
            return (prev_sample,prev_p)

        return SGHMCSchedulerOutput(prev_sample=prev_sample, 
                                    momentum=prev_p,
                                    pred_original_sample=pred_original_sample)

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps
