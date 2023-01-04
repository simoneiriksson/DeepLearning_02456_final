from utils.data_transformers import *
from torch import nn, Tensor
import torch
from typing import List, Set, Dict, Tuple, Optional, Any
import math


def reduce(x:Tensor) -> Tensor:
    """for each datapoint: sum over all dimensions"""
    flat = view_flat_samples(x)
    return flat.sum(dim=1)

class VariationalInference_VAE(nn.Module):
    def __init__(self, beta:float=1., p_norm = 2.):
        super().__init__()
        self.beta = beta
        self.p_norm = float(p_norm)

    def update_vi(self):
        pass
            
    def forward(self, model:nn.Module, x:Tensor) -> Tuple[Tensor, Dict]:
        outputs = model(x)

        # Unpack values from VAE
        x_hat, qz_log_sigma, qz_mu, z = [outputs[k] for k in ["x_hat", "qz_log_sigma", "qz_mu", "z"]]
        qz_sigma = qz_log_sigma.exp()
        # Imagewise loss. Calculated as the p-norm distance in pixel-space between original and reconstructed image
        image_loss = ((x_hat - x).abs()**self.p_norm).sum(axis=[1,2,3])

        # KL-divergence calculated explicitly
        # Reference Kingma & Welling p. 5 bottom
        kl = - (.5 * (1 + (qz_sigma ** 2).log() - qz_mu ** 2 - qz_sigma**2)).sum(axis=[1])

        # Image-wise beta-elbo:
        beta_elbo = -image_loss - self.beta * kl

        # Loss is the mean of the imagewise losses, over the full batch of images
        loss = -beta_elbo.mean()
        
        # prepare the output
        with torch.no_grad():
            diagnostics = {'elbo': beta_elbo, 'image_loss':image_loss, 'kl': kl}
            
        return loss, diagnostics, outputs
      
