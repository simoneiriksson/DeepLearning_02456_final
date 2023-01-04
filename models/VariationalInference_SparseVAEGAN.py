# build inference class as detailed above. 

from utils.data_transformers import *
from torch import nn, Tensor
import torch
from typing import List, Set, Dict, Tuple, Optional, Any
import math


class VariationalInference_SparseVAEGAN(nn.Module):
    def __init__(self, p_norm = 2., beta:float=1., alpha:float=0.0):
        super().__init__()
        self.p_norm = float(p_norm)
        self.beta = beta
        self.alpha = alpha
    def update_vi(self):
        pass
            
    def forward(self, VAE_model:nn.Module, DISC, x:Tensor) -> Tuple[Tensor, Dict]:
        outputs = VAE_model(x)

        # Unpack values from VAE
        x_hat, z, qz_log_gamma, qz_mu, qz_log_sigma = [outputs[k] for k in ['x_hat', 'z', 'qz_log_gamma', 'qz_mu', 'qz_log_sigma']]
        qz_sigma = qz_log_sigma.exp()
        #print("qz_sigma.shape: ", qz_sigma.shape)
        
        _, disc_repr_x = DISC(x)
        _, disc_repr_x_hat = DISC(x_hat)
        outputs['disc_repr_x'] = disc_repr_x
        outputs['disc_repr_x_hat'] = disc_repr_x_hat

        # KL-divergence calculated explicitly
        qz_gamma = torch.clamp(qz_log_gamma.exp(), 1e-6, 1.0 - 1e-6) 
        KL_part1 = 0.5 * qz_gamma.mul(1 + qz_log_sigma * 2 - qz_mu ** 2 - qz_log_sigma.exp() ** 2)
        KL_part2 = (1 - qz_gamma).mul(((1 - self.alpha)/(1 - qz_gamma)).log())
        KL_part3 = qz_gamma.mul((self.alpha/qz_gamma).log())
        kl_div = -(KL_part1 + KL_part2 + KL_part3).sum(axis=[1])

        # Imagewise loss. Calculated as the p-norm distance in pixel-space between original and reconstructed image
        image_loss = ((x_hat - x).abs()**self.p_norm).sum(axis=[1,2,3])

        # Discriminator representation loss
        flat_disc_repr_x = [None]*len(disc_repr_x)
        flat_disc_repr_x_hat = [None]*len(disc_repr_x_hat)
        for i in range(len(disc_repr_x)):
            flat_disc_repr_x[i] = torch.flatten(disc_repr_x[i], start_dim=1, end_dim=3)
            flat_disc_repr_x_hat[i] = torch.flatten(disc_repr_x_hat[i], start_dim=1, end_dim=3)
        flattened_disc_repr_x = torch.cat(flat_disc_repr_x, dim=1)

        flattened_disc_repr_x_hat = torch.cat(flat_disc_repr_x_hat, dim=1)

        disc_repr_loss = ((flattened_disc_repr_x_hat - flattened_disc_repr_x).abs()**2).sum(axis=[1])

        # Discriminator loss
        disc_x_d, _ = DISC(x.detach())
        disc_x_hat_d, _ = DISC(x_hat.detach())
        outputs['disc_x_d'] = disc_x_d
        outputs['disc_x_hat_d'] = disc_x_hat_d
        
        outputs['disc_real_pred'] = disc_x_d.round()
        outputs['disc_fake_pred'] = disc_x_hat_d.round()        
        
        disc_groundtruth = torch.concat((torch.ones_like(disc_x_d, ), torch.zeros_like(disc_x_hat_d)))
        disc_cat = torch.concat((disc_x_d, disc_x_hat_d))
        
        disc_loss = - disc_groundtruth.mul(disc_cat.log()) - (1-disc_groundtruth).mul((1-disc_cat).log())

        # prepare the output
        image_loss_mean = image_loss.mean()
        kl_div_mean = kl_div.mean()
        disc_loss_mean = disc_loss.mean()
        disc_repr_loss_mean = disc_repr_loss.mean()

        with torch.no_grad():
            losses_mean = {
                'image_loss': image_loss_mean,
                'kl_div': kl_div_mean,
                'disc_loss': disc_loss_mean,
                'disc_repr_loss': disc_repr_loss_mean
                }
            losses = {
                'image_loss': image_loss,
                'kl_div': kl_div,
                'disc_loss': disc_loss,
                'disc_repr_loss': disc_repr_loss
                }
        return losses_mean, losses, outputs
      
