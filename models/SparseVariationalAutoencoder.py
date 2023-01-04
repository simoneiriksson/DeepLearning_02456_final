import numpy as np
from torch import nn, Tensor
import torch
from torch.distributions import Distribution, Exponential, Cauchy, HalfCauchy, Normal
#from models.PrintSize import PrintSize
from typing import List, Set, Dict, Tuple, Optional, Any


class SparseVariationalAutoencoder(nn.Module):
    def ReparameterizedSpikeAndSlab_sample(self, mu, log_sigma, log_gamma):
        eps = torch.empty_like(log_sigma.exp()).normal_()
        eta = torch.empty_like(log_sigma.exp()).normal_()
        selector = torch.sigmoid((log_gamma.exp() + eta -1))    
        return selector * (mu + eps.mul(log_sigma.exp()))


    def __init__(self, input_shape, latent_features: int) -> None:
        super(SparseVariationalAutoencoder, self).__init__()
        #print("Init SVAE input_shape, latent_features: ", input_shape, latent_features)
        self.input_shape = getattr(input_shape, "tolist", lambda: input_shape)()
        #print("Init SVAE self.input_shape: ", tuple(self.input_shape))
        
        self.latent_features = latent_features
        self.observation_features = np.prod(input_shape)
        self.input_channels = input_shape[0]
        self.serect_c = 50


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
            nn.Conv2d(in_channels=32, out_channels=3*latent_features, kernel_size=5, padding=0),
            # Now we are at: 1h * 1w * 512ch
            nn.BatchNorm2d(3*self.latent_features ),
            nn.Flatten()
        )

        
        # Generative Model
        # Decode the latent sample `z` into the parameters of the observation model
        # `p_\theta(x | z) = \prod_i B(x_i | g_\theta(x))`
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (self.latent_features ,1,1)), # Now we are at: 1h * 1w * 256ch
            nn.Conv2d(in_channels=self.latent_features , out_channels=32, kernel_size=5, padding=4),
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
        self.register_buffer('prior_params', torch.zeros(torch.Size([1, 2*latent_features])))
        #print("Init SVAE end")

    def observation(self, z:Tensor) -> Tensor:
        """return the distribution `p(x|z)`"""
        mu = self.decoder(z)
        mu = mu.view(-1, *self.input_shape) # reshape the output
        return mu
    
    def forward(self, x) -> Dict[str, Any]:
        # flatten the input
        #x = x.reshape(x.size(0), -1)
        h_z = self.encoder(x)
        qz_mu, qz_log_sigma, qz_log_gamma = h_z.chunk(3, dim=-1)

        z = self.ReparameterizedSpikeAndSlab_sample(qz_mu, qz_log_sigma, qz_log_gamma)
        x_hat = self.observation(z)
        #print("x_hat.shape", x_hat.shape) 
        
        return {'x_hat': x_hat, 
                'z': z, 
                'qz_log_gamma': qz_log_gamma, 
                'qz_mu': qz_mu, 
                'qz_log_sigma':qz_log_sigma}

    def update_(self):
        self.serect_c += 0
        

    