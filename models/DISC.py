import numpy as np
from torch import nn, Tensor
import torch
from models.PrintSize import PrintSize
from typing import List, Set, Dict, Tuple, Optional, Any


class MyClamp(nn.Module):
    def forward(self, x):
        return torch.clamp(x, -10, 10)


class DISC(nn.Module):

    def __init__(self, input_shape, latent_features: int) -> None:
        super(DISC, self).__init__()
        self.input_shape = getattr(input_shape, "tolist", lambda: input_shape)()        
        self.latent_features = latent_features
        self.observation_features = np.prod(input_shape)
        self.input_channels = input_shape[0]

        self.level0 = nn.Sequential(
            PrintSize(),
            # now we are at 68h * 68w * 3ch
            nn.Conv2d(in_channels=self.input_channels, out_channels=32, kernel_size=5, padding=0),
            # Now we are at: 64h * 64w * 32ch
            nn.MaxPool2d(2),
            nn.LeakyReLU(negative_slope=0.01))

        self.level1 = nn.Sequential(
            PrintSize(),
            nn.BatchNorm2d(32),
            # Now we are at: 32h * 32w * 32ch
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=0),
            # Now we are at: 28h * 28w * 32ch
            nn.MaxPool2d(2),
            # Now we are at: 14h * 14w * 32ch
            nn.LeakyReLU(negative_slope=0.01))

        self.level2 = nn.Sequential(
            PrintSize(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=0),
            # Now we are at: 10h * 10w * 32ch
            nn.MaxPool2d(2),
            # Now we are at: 5h * 5w * 32ch
            nn.LeakyReLU(negative_slope=0.01)
            )

        self.level3 = nn.Sequential(
            PrintSize(),
            nn.BatchNorm2d(32),
            ##Output should be 5*5*32 now.
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=0),
            # Now we are at: 1h * 1w * 64ch
            # 64 ch out is like they do in Lafarge code
            nn.BatchNorm2d(64))

        self.level4 = nn.Sequential(
            PrintSize(),
            #nn.Flatten(),
            PrintSize(),
            nn.LeakyReLU(negative_slope=0.01),
            #nn.Linear(in_features=64, out_features=1),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, padding=0),
            PrintSize(),
            nn.Flatten(),
            MyClamp(),
            nn.Sigmoid()
        )

    def forward(self, x) -> Dict[str, Any]:
        latents = [None] * 4
        latents[0] = self.level0(x)
        latents[1] = self.level1(latents[0])
        latents[2] = self.level2(latents[1])
        latents[3] = self.level3(latents[2])
        Discrim = self.level4(latents[3])
        

        return Discrim, latents

    def update_(self):
        pass
