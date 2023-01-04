# build inference class as detailed above. 

from utils.data_transformers import *
from torch import nn, Tensor
import torch
from typing import List, Set, Dict, Tuple, Optional, Any
import math


class VariationalInference_VAEGAN(nn.Module):
    def __init__(self, p_norm = 2., beta:float=1.):
        super().__init__()
        self.beta = beta
        self.p_norm = float(p_norm)

    def update_vi(self):
        pass
            
    def forward(self, VAE_model:nn.Module, DISC, x:Tensor) -> Tuple[Tensor, Dict]:
        outputs = VAE_model(x)

        # Unpack values from VAE
        x_hat, qz_log_sigma, qz_mu, z = [outputs[k] for k in ["x_hat", "qz_log_sigma", "qz_mu", "z"]]
        qz_sigma = qz_log_sigma.exp()
        #print("qz_sigma.shape: ", qz_sigma.shape)
        
        _, disc_repr_x = DISC(x)
        _, disc_repr_x_hat = DISC(x_hat)
        outputs['disc_repr_x'] = disc_repr_x
        outputs['disc_repr_x_hat'] = disc_repr_x_hat

        # KL-divergence calculated explicitly
        kl_div = - (.5 * (1 + (qz_sigma ** 2).log() - qz_mu ** 2 - qz_sigma**2)).sum(axis=[1])

        # Imagewise loss. Calculated as the p-norm distance in pixel-space between original and reconstructed image
        image_loss = ((x_hat - x).abs()**self.p_norm).sum(axis=[1,2,3])

        # Discriminator representation loss
        flat_disc_repr_x = [None]*len(disc_repr_x)
        flat_disc_repr_x_hat = [None]*len(disc_repr_x_hat)
        for i in range(len(disc_repr_x)):
            #print(f"disc_repr_x[{i}].shape: {disc_repr_x[i].shape}")
            flat_disc_repr_x[i] = torch.flatten(disc_repr_x[i], start_dim=1, end_dim=3)
            #print(f"flat_disc_repr_x[{i}].shape: {flat_disc_repr_x[i].shape}")
            flat_disc_repr_x_hat[i] = torch.flatten(disc_repr_x_hat[i], start_dim=1, end_dim=3)
        flattened_disc_repr_x = torch.cat(flat_disc_repr_x, dim=1)

        #print(f"flattened_disc_repr_x.shape: {flattened_disc_repr_x.shape}")
        flattened_disc_repr_x_hat = torch.cat(flat_disc_repr_x_hat, dim=1)

        #print(f"flattened_disc_repr_x_hat: {flattened_disc_repr_x_hat}")
        #print(f"flattened_disc_repr_x: {flattened_disc_repr_x}")
        #disc_repr_loss = torch.nn.functional.mse_loss(flattened_disc_repr_x, flattened_disc_repr_x_hat)
        disc_repr_loss = ((flattened_disc_repr_x_hat - flattened_disc_repr_x).abs()**2).sum(axis=[1])
        #print("disc_repr_loss: ", disc_repr_loss)

        # Discriminator loss
        disc_x_d, _ = DISC(x.detach())
        disc_x_hat_d, _ = DISC(x_hat.detach())
        outputs['disc_x_d'] = disc_x_d
        outputs['disc_x_hat_d'] = disc_x_hat_d
        
        outputs['disc_real_pred'] = disc_x_d.round()
        outputs['disc_fake_pred'] = disc_x_hat_d.round()        
        
        disc_groundtruth = torch.concat((torch.ones_like(disc_x_d, ), torch.zeros_like(disc_x_hat_d)))
        disc_cat = torch.concat((disc_x_d, disc_x_hat_d))
        #print("disc_groundtruth.shape: ", disc_groundtruth.shape)
        #print("disc_cat.shape: ", disc_cat.shape)
        
        #disc_loss = self.DiscLoss_fn(disc_cat, disc_groundtruth)
        disc_loss = - disc_groundtruth.mul(disc_cat.log()) - (1-disc_groundtruth).mul((1-disc_cat).log())
        #print("disc_loss.shape: ", disc_loss.shape)
        #print("disc_loss: ", disc_loss)

        # prepare the output
        image_loss_mean = image_loss.mean()
        kl_div_mean = kl_div.mean()
        disc_loss_mean = disc_loss.mean()
        disc_repr_loss_mean = disc_repr_loss.mean()

        #print("image_loss_mean: ", image_loss_mean)
        #print("kl_div_mean: ", kl_div_mean)
        #print("disc_loss_mean: ", disc_loss_mean)
        #print("disc_repr_loss_mean: ", disc_repr_loss_mean)
        

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
      
