from diffusers import DiffusionPipeline
import torch.nn as nn
import torch
from torch.cuda.amp import custom_bwd, custom_fwd
import pdb
from safetensors.torch import load_file
from einops import rearrange, repeat
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from scipy.spatial import Delaunay
import numpy as np
from torch.nn import functional as F
import os

# =============================================
# ===== Helper function for SDS gradients =====
# =============================================
class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None


# ========================================================
# ===== Basic class to extend with SDS loss variants =====
# ========================================================
class SDSLossBase(nn.Module):
    # global diffusion pipeline, used to share between different instances
    _global_pipe = None

    def __init__(self, cfg, device, reuse_pipe=True):
        """
        Initialize SDS loss base class
        Args:
            cfg: configuration parameters
            device: running device
            reuse_pipe: whether to reuse existing diffusion pipeline
        """
        super(SDSLossBase, self).__init__()

        self.cfg = cfg
        self.device = device

        self.maybe_init_pipe(reuse_pipe) 

        self.alphas = self.pipe.scheduler.alphas_cumprod.to(self.device)
        self.sigmas = (1 - self.pipe.scheduler.alphas_cumprod).to(self.device)

        if cfg.use_xformers:
            self.pipe.enable_xformers_memory_efficient_attention()

        self.text_embeddings = self.embed_text(self.cfg.caption)

        if self.cfg.del_text_encoders:
            del self.pipe.tokenizer
            del self.pipe.text_encoder

    def maybe_init_pipe(self, reuse_pipe):
        """初始化diffusion pipeline
        Args:
            reuse_pipe: 是否重用已有pipeline
        """
        if reuse_pipe:
            if SDSLossBase._global_pipe is None:
                SDSLossBase._global_pipe = DiffusionPipeline.from_pretrained(self.cfg.model_name, torch_dtype=torch.float16, variant="fp16", local_files_only=True, use_safetensors=True, load_in_8bit=False)
                SDSLossBase._global_pipe = SDSLossBase._global_pipe.to(self.device)
            self.pipe = SDSLossBase._global_pipe
        else:
            self.pipe = DiffusionPipeline.from_pretrained(self.cfg.model_name, torch_dtype=torch.float16, variant="fp16", local_files_only=True)
            self.pipe = self.pipe.to(self.device)

    def embed_text(self, caption):
        text_input = self.pipe.tokenizer(caption, padding="max_length",
                                         max_length=self.pipe.tokenizer.model_max_length,
                                         truncation=True, return_tensors="pt")
        # for unconditional generation
        uncond_input = self.pipe.tokenizer([""], padding="max_length",
                                         max_length=text_input.input_ids.shape[-1],
                                         return_tensors="pt")
        with torch.no_grad():
            text_embeddings = self.pipe.text_encoder(text_input.input_ids.to(self.device))[0]
            uncond_embeddings = self.pipe.text_encoder(uncond_input.input_ids.to(self.device))[0]
            
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        text_embeddings = text_embeddings.repeat_interleave(self.cfg.batch_size, 0)

        return text_embeddings

        
    def prepare_latents(self, x_aug):
        x = x_aug * 2. - 1.
        
        with torch.cuda.amp.autocast():
            batch_size, num_frames, channels, height, width = x.shape # [1, K, 3, 256, 256]
            x = x.reshape(batch_size * num_frames, channels, height, width) # [K, 3, 256, 256]
            init_latent_z = (self.pipe.vae.encode(x).latent_dist.sample()) # [K, 4, 32, 32]
            frames, channel, h_, w_ = init_latent_z.shape
            init_latent_z = init_latent_z[None, :].reshape(batch_size, num_frames, channel, h_, w_).permute(0, 2, 1, 3, 4) # [1, 4, K, 32, 32]
            
        latent_z = self.pipe.vae.config.scaling_factor * init_latent_z

        return latent_z

    def add_noise_to_latents(self, latent_z, timestep, return_noise=True, eps=None):
        if eps is None:
            if self.cfg.same_noise_for_frames: 
                eps = torch.randn_like(latent_z[:, :, 0, :, :]) 
                eps = repeat(eps, 'b c h w -> b c f h w', f=latent_z.shape[2])
            else:
                eps = torch.randn_like(latent_z)

        noised_latent_zt = self.pipe.scheduler.add_noise(latent_z, eps, timestep)

        if return_noise:
            return noised_latent_zt, eps

        return noised_latent_zt
    
    def get_sds_eps_to_subract(self, eps_orig, z_in, timestep_in):
        return eps_orig

    def drop_nans(self, grads):
        assert torch.isfinite(grads).all()
        return torch.nan_to_num(grads.detach().float(), 0.0, 0.0, 0.0)

    def get_grad_weights(self, timestep):
        return (1 - self.alphas[timestep])

    def sds_grads(self, latent_z, **sds_kwargs):

        with torch.no_grad():
            timestep = torch.randint(
                low=self.cfg.sds_timestep_low,
                high=min(950, self.cfg.timesteps) - 1,  # avoid highest timestep | diffusion.timesteps=1000
                size=(latent_z.shape[0],),
                device=self.device, dtype=torch.long)

            noised_latent_zt, eps = self.add_noise_to_latents(latent_z, timestep, return_noise=True)

            z_in = torch.cat([noised_latent_zt] * 2)  
            timestep_in = torch.cat([timestep] * 2)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                eps_t_uncond, eps_t = self.pipe.unet(z_in, timestep_in, encoder_hidden_states=self.text_embeddings).sample.float().chunk(2)

            eps_t = eps_t_uncond + self.cfg.guidance_scale * (eps_t - eps_t_uncond)

            eps_to_subtract = self.get_sds_eps_to_subract(eps, z_in, timestep_in, **sds_kwargs)

            w = self.get_grad_weights(timestep)
            grad_z = w * (eps_t - eps_to_subtract)

            grad_z = self.drop_nans(grad_z)

        return grad_z
    
    


# =======================================
# =========== Basic SDS loss  ===========
# =======================================
class SDSVideoLoss(SDSLossBase):
    def __init__(self, cfg, device, reuse_pipe=True):
        super(SDSVideoLoss, self).__init__(cfg, device, reuse_pipe=reuse_pipe)

    def forward(self, x_aug, grad_scale=1.0):

        latent_z = self.prepare_latents(x_aug)

        grad_z = grad_scale * self.sds_grads(latent_z)

        sds_loss = SpecifyGradient.apply(latent_z, grad_z)

        return sds_loss    