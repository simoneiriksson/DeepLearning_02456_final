from typing import List, Set, Dict, Tuple, Optional, Any
from collections import defaultdict
import torch
import numpy as np

from models.CytoVariationalAutoencoder import CytoVariationalAutoencoder
from models.DISC import DISC
from models.SparseVariationalAutoencoder import SparseVariationalAutoencoder
from models.VariationalInference_VAE import VariationalInference_VAE
from models.VariationalInference_VAEGAN import VariationalInference_VAEGAN
from models.VariationalInference_SparseVAEGAN import VariationalInference_SparseVAEGAN
from models.VariationalInference_SparseVAE import VariationalInference_SparseVAE
from utils.utils import cprint

def LoadVAEmodel(folder, model_type=None, device="cpu"):
    params = torch.load(folder + "params.pt", map_location=torch.device(device))

    validation_data = torch.load(folder + "validation_data.pt", map_location=torch.device(device))
    training_data = torch.load(folder + "training_data.pt", map_location=torch.device(device))
    
    model_type = params['model_type']
    if 'p_norm' in params.keys(): p_norm = params['p_norm'] 
    else: p_norm = 2

    if model_type in ['Cyto_nonvar', 'CytoVAE']:
        vae = CytoVariationalAutoencoder(params['image_shape'], params['latent_features'])
        vae.load_state_dict(torch.load(folder + "vae_parameters.pt", map_location=torch.device(device)))
        vi = VariationalInference_VAE(beta=params['beta'], p_norm = p_norm)
        model = [vae]

    if model_type in ['Cyto_VAEGAN', 'CytoVAEGAN']:
        vae = CytoVariationalAutoencoder(params['image_shape'], params['latent_features'])
        disc = DISC(params['image_shape'], params['latent_features'])
        vae.load_state_dict(torch.load(folder + "vae_parameters.pt", map_location=torch.device(device)))
        disc.load_state_dict(torch.load(folder + "disc_parameters.pt", map_location=torch.device(device)))
        model = [vae, disc]
        vi = VariationalInference_VAEGAN(beta=params['beta'], p_norm = p_norm)

    if model_type == 'SparseVAEGAN':
        vae = SparseVariationalAutoencoder(params['image_shape'], params['latent_features'])
        disc = DISC(params['image_shape'], params['latent_features'])
        vae.load_state_dict(torch.load(folder + "vae_parameters.pt", map_location=torch.device(device)))
        disc.load_state_dict(torch.load(folder + "disc_parameters.pt", map_location=torch.device(device)))
        model = [vae, disc]
        vi = VariationalInference_SparseVAEGAN(beta=params['beta'], alpha=params['alpha'], p_norm = p_norm)

    if model_type == 'SparseVAE':
        vae = SparseVariationalAutoencoder(params['image_shape'], params['latent_features'])
        vae.load_state_dict(torch.load(folder + "vae_parameters.pt", map_location=torch.device(device)))
        vi = VariationalInference_SparseVAE(beta=params['beta'], alpha=params['alpha'], p_norm = p_norm)
        model = [vae]
    return model, validation_data, training_data, params, vi

# Change model loader to return GAN/VAE couple
def initVAEmodel(params):

    model_type = params['model_type']

    training_performance = defaultdict(list)
    validation_performance = defaultdict(list)
    if 'p_norm' in params.keys(): p_norm = params['p_norm'] 
    else: p_norm = 2

    if model_type in ['Cyto_nonvar', 'CytoVAE']:
        vae = CytoVariationalAutoencoder(params['image_shape'], params['latent_features'])
        vi = VariationalInference_VAE(beta=params['beta'], p_norm = p_norm)
        model = [vae]

    elif model_type == 'CytoVAEGAN':
        vae = CytoVariationalAutoencoder(params['image_shape'], params['latent_features'])
        disc = DISC(params['image_shape'], params['latent_features'])
        model = [vae, disc]
        vi = VariationalInference_VAEGAN(beta=params['beta'], p_norm = p_norm)

    elif model_type == 'SparseVAEGAN':
        vae = SparseVariationalAutoencoder(params['image_shape'], params['latent_features'])
        disc = DISC(params['image_shape'], params['latent_features'])
        model = [vae, disc]
        vi = VariationalInference_SparseVAEGAN(beta=params['beta'], alpha=params['alpha'], p_norm = p_norm)

    elif model_type == 'SparseVAE':
        vae = SparseVariationalAutoencoder(params['image_shape'], params['latent_features'])
        vi = VariationalInference_SparseVAE(beta=params['beta'], alpha=params['alpha'], p_norm = p_norm)
        model = [vae]
    else:
        cprint(f"incorrect model_type: {model_type}")  
    return model, validation_performance, training_performance, params, vi

