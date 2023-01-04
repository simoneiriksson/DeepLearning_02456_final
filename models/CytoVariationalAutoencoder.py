import numpy as np
from torch import nn, Tensor
import torch
#from models.PrintSize import PrintSize
from typing import List, Set, Dict, Tuple, Optional, Any

class CytoVariationalAutoencoder(nn.Module):
 
    def update_(self):
        pass
  
    def __init__(self, input_shape, latent_features: int):
        super(CytoVariationalAutoencoder, self).__init__()
        
        self.input_shape = input_shape
        self.latent_features = latent_features
        self.observation_features = np.prod(input_shape)
        self.observation_shape = input_shape
        self.input_channels = input_shape[0]
        self.epsilon = 10e-3
        
        # Inference Network
        # Encode the observation `x` into the parameters of the posterior distribution
        # `q_\phi(z|x) = N(z | \mu(x), \sigma(x)), \mu(x),\log\sigma(x) = h_\phi(x)`
        self.encoder = nn.Sequential(
            # now we are at 68h * 68w * 3ch
            nn.Conv2d(in_channels=self.input_channels, out_channels=32, kernel_size=5, padding=0),
            # Now we are at: 64h * 64w * 32ch
            nn.MaxPool2d(2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),

            # Now we are at: 32h * 32w * 32ch
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=0),
            # Now we are at: 28h * 28w * 32ch
            nn.MaxPool2d(2),
            # Now we are at: 14h * 14w * 32ch
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=0),
            # Now we are at: 10h * 10w * 32ch
            nn.MaxPool2d(2),
            # Now we are at: 5h * 5w * 32ch
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),

            ##Output should be 5*5*32 now.
            nn.Conv2d(in_channels=32, out_channels=2*self.latent_features, kernel_size=5, padding=0),
            # Now we are at: 1h * 1w * 512ch
            nn.BatchNorm2d(2*self.latent_features),
            nn.Flatten()
        )

        
        # Generative Model
        # Decode the latent sample `z` into the parameters of the observation model
        # `p_\theta(x | z) = \prod_i B(x_i | g_\theta(x))`
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (self.latent_features,1,1)), # Now we are at: 1h * 1w * 256ch
            nn.Conv2d(in_channels=self.latent_features, out_channels=32, kernel_size=5, padding=4),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01),
            torch.nn.UpsamplingNearest2d(size=10),

            # Now we are at: 10h * 10w * 32ch
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=4),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01),
            torch.nn.UpsamplingNearest2d(size=28),

            # Now we are at: 28h * 28w * 32ch
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=4),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01),
            torch.nn.UpsamplingNearest2d(size=64),

            # Now we are at: 64h * 64w * 32ch
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=4),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01),
            
            # Now we are at: 68h * 68w * 32ch
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1, padding=0), # 6 channels because 3 for mean and 3 for variance
#            nn.BatchNorm2d(6),
            nn.LeakyReLU(negative_slope=0.01)
        )
        
    def observation(self, z:Tensor) -> Tensor:
        """return the distribution `p(x|z)`"""
        #h_z = self.decoder(z)
        #mu, log_sigma = h_z.chunk(2, dim=-1)
        mu = self.decoder(z)
        mu = mu.view(-1, *self.input_shape) # reshape the output
        #log_sigma = log_sigma.view(-1, *self.input_shape) # reshape the output
        return mu

    def forward(self, x) -> Dict[str, Any]:
        """compute the posterior q(z|x) (encoder), sample z~q(z|x) and return the distribution p(x|z) (decoder)"""
        
        h_z = self.encoder(x)
        qz_mu, qz_log_sigma =  h_z.chunk(2, dim=-1)        
        eps = torch.empty_like(qz_mu).normal_()
        z = qz_mu + qz_log_sigma.exp() * eps
        
        x_hat = self.observation(z)
        
        return {'x_hat': x_hat, 'qz_log_sigma': qz_log_sigma, 'qz_mu': qz_mu, 'z': z}
    
    
