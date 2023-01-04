from typing import List, Set, Dict, Tuple, Optional, Any
from collections import defaultdict
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import math 
import torch
from torch import nn, Tensor
from torch.nn.functional import softplus, relu
from torch.distributions import Distribution, Normal
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Dataset

from models.LoadModels import LoadVAEmodel, initVAEmodel
from utils.data_transformers import normalize_every_image_channels_seperately_inplace
from utils.data_transformers import normalize_channels_inplace, batch_normalize_images
from utils.data_transformers import SingleCellDataset
from utils.plotting import plot_VAE_performance, plot_image_channels, extract_a_few_images
from utils.data_preparation import create_directory, read_metadata, get_relative_image_paths, load_images
from utils.data_preparation import read_metadata_and_images
from utils.data_preparation import get_MOA_mappings, shuffle_metadata, split_metadata
from utils.utils import cprint, get_datetime, create_logfile, constant_seed, StatusString
from utils.utils import save_model
from utils.profiling import LatentVariableExtraction
#from utils.plotting import heatmap, plot_cosine_similarity
from utils.profiling import NSC_NearestNeighbor_Classifier, moa_confusion_matrix, Accuracy
from downstream_task import downstream_task

import importlib

from VAE_trainer import VAE_trainer 
from VAEGAN_trainer import VAEGAN_trainer

######### Utilities #########
# choose correct output folder for LoadVAEmodel() below!!!
#output_folder = "./dump/outputs_2022-12-28 - 09-40-32/"
#output_folder = "./dump/outputs_2022-12-28 - 09-53-14/"
#output_folder = "./dump/outputs_2023-01-01 - 22-24-26/"

#output_folder = "./dump/outputs_2022-12-31 - 09-23-19_SparseVAE_beta1_alpha0.05_epochs100/"
#output_folder = "./dump/outputs_2022-12-30 - 21-23-27_Cyto_nonvar_beta0_epochs100/"
#output_folder = "./dump/outputs_2022-12-29 - 20-21-46-20230101T183446Z-001/"
#output_folder = "./dump/outputs_2022-12-31 - 14-22-48_SparseVAE_beta1_alpha0.2_epochs100/"
#output_folder = "./dump/outputs_2023-01-02 - 18-47-44/"
#output_folder = "./dump/outputs_2023-01-02 - 19-35-44/"
#output_folder = "./dump/outputs_2023-01-02 - 19-35-10/"
#output_folder = "./dump/outputs_2023-01-02 - 20-02-52_CytoVAE_beta0_epochs50/"
#output_folder = "./dump/outputs_2023-01-02 - 20-10-36_SparseVAE_beta1_alpha0.05_epochs50/"
#output_folder = "./dump/outputs_2023-01-02 - 20-08-32_SparseVAE_beta1_alpha0.2_epochs50/"

#output_folder = "./dump/outputs_2023-01-02 - 20-20-03_SparseVAEGAN_beta1_alpha0.05_epochs50/"
#output_folder = "./dump/outputs_2023-01-02 - 20-22-11/"
#output_folder = "./dump/outputs_2023-01-02 - 19-38-25/"

#output_folder = "./dump/"
#####
#output_folder = "./dump/outputs_2023-01-02 - 19-35-44/" 
output_folder = "./dump/outputs_2023-01-02 - 19-35-10/" 
#output_folder = "./dump/outputs_2023-01-02 - 21-17-12/"
#output_folder = "./dump/outputs_2023-01-02 - 18-47-44/"

constant_seed()
logfile = create_logfile(output_folder + "downstream_log.log")
cprint("output_folder is: {}".format(output_folder), logfile)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cprint(f"Using device: {device}", logfile)

#images, metadata, metadata_all, mapping = read_metadata_and_images(use_server_path = True, \
#                                                        load_images_from_individual_files = False, 
#                                                        load_subset_of_images = None, 
#                                                        save_images_to_singlefile = False,
#                                                        shuffle = False,
#                                                        logfile = logfile)

# Settings for handing in:
images, metadata, mapping = read_metadata_and_images(use_server_path = True, \
                                                        load_images_from_individual_files = True, 
                                                        load_subset_of_images = None, 
                                                        save_images_to_singlefile = False,
                                                        shuffle = False,
                                                        logfile = logfile)
                                                        )

# With the below command, we normalize all the images, image- and channel-wise.
# Alternative, this can be uncommented and like in the Lafarge article, we can do batchwise normalization
normalize_every_image_channels_seperately_inplace(images, verbose=True)

metadata_train, metadata_validation = split_metadata(metadata, split_fraction = .90)

train_set = SingleCellDataset(metadata_train, images, mapping)
validation_set = SingleCellDataset(metadata_validation, images, mapping)

#### LOAD TRAINED MODEL ####
model, validation_data, training_data, params, vi = LoadVAEmodel(output_folder)

cprint("model is of type {}".format(params['model_type']), logfile)
cprint("model parameters are: {}".format(params), logfile)

vae = model[0]

if params['model_type'] in ['SparseVAEGAN', 'CytoVAEGAN']:
    gan = model[1]

train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True, num_workers=0, drop_last=True)
validation_loader = DataLoader(validation_set, batch_size=max(2, params['batch_size']), shuffle=False, num_workers=0, drop_last=False)

_ = vae.eval() # because of batch normalization

downstream_task(vae, metadata, images, mapping, device, output_folder, logfile)
    
cprint("output_folder is: {}".format(output_folder), logfile)
cprint("script done.", logfile)

