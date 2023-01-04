#from utils.data_transformers import *
from torch import nn, Tensor
import torch
from typing import List, Set, Dict, Tuple, Optional, Any
import math


class VariationalInference_SparseVAE(nn.Module):
    def __init__(self, p_norm = 2., beta:float=1., alpha:float=0.0):
        super().__init__()
        self.beta = beta
        self.alpha = alpha        
        self.p_norm = float(p_norm)

    def update_vi(self):
        pass

    def forward(self, model:nn.Module, x:Tensor) -> Tuple[Tensor, Dict]:
        outputs = model(x)

        x_hat, z, qz_log_gamma, qz_mu, qz_log_sigma = [outputs[k] for k in ['x_hat', 'z', 'qz_log_gamma', 'qz_mu', 'qz_log_sigma']]
        
        # My implementation
        #qz_gamma = qz_log_gamma.exp()
        #qz_gamma = torch.clamp(qz_log_gamma.exp(), 1e-6, 1.0 - 1e-6) 
        #KL_part1 = qz_gamma.mul(1 + qz_log_sigma * 2 - qz_mu ** 2 - qz_log_sigma.exp() ** 2)/2
        #KL_part2 = -(1 - qz_gamma).mul(((1 - self.alpha).div(1 - qz_gamma)).log())
        #KL_part3 = -qz_gamma.mul((self.alpha.div(qz_gamma)).log())
        
        # implementation adapted from article
        qz_gamma = torch.clamp(qz_log_gamma.exp(), 1e-6, 1.0 - 1e-6) 
        KL_part1 = 0.5 * qz_gamma.mul(1 + qz_log_sigma * 2 - qz_mu ** 2 - qz_log_sigma.exp() ** 2)
        KL_part2 = (1 - qz_gamma).mul(((1 - self.alpha)/(1 - qz_gamma)).log())
        KL_part3 = qz_gamma.mul((self.alpha/qz_gamma).log())
    
    
        KL = -(KL_part1 + KL_part2 + KL_part3).sum(axis=[1])
        
        mse_loss = ((x_hat - x)**self.p_norm).sum(axis=[1,2,3])

        beta_elbo = -self.beta * KL - mse_loss

        loss = -beta_elbo.mean()
        
        # prepare the output
        with torch.no_grad():
            diagnostics = {'elbo': beta_elbo, 'mse_loss':mse_loss, 'kl': KL}
            
        return loss, diagnostics, outputs