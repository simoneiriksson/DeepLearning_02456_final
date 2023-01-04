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

from utils.plotting import plot_VAE_performance, plot_image_channels, extract_a_few_images
from utils.data_preparation import create_directory, read_metadata, get_relative_image_paths, load_images
from utils.utils import cprint
from utils.profiling import LatentVariableExtraction
from utils.plotting import heatmap
from utils.profiling import NSC_NearestNeighbor_Classifier, moa_confusion_matrix, Accuracy, precision, recall
from utils.profiling import treatment_profiles, treatment_center_cells
from utils.plotting import plot_control_cell_to_target_cell
        
def downstream_task(vae, metadata, images, mapping, device, output_folder, logfile=None):
    cprint("Starting downstream tasks", logfile)
    #device = 'cpu'
    vae = vae.to(device)

    _ = vae.eval() # because of batch normalization

    cprint("Extract a few images", logfile)
    extract_a_few_images(output_folder + "images", vae=vae, no_images=10, dataset=images, device=device, logfile=logfile)
    cprint("saved images", logfile)

    #### CALCULATE LATENT REPRESENTATION FOR ALL IMAGES ####
    cprint("Calculate latent representation for all images", logfile)
    batch_size= 1024
    metadata_latent = LatentVariableExtraction(metadata, images, batch_size, vae, device, logfile)
    cprint("Done calculating latent space", logfile)

    
    cprint("Plotting interpolations of reconstructions", logfile)
    create_directory(output_folder + "interpolations")
    #treatments list
    tl = metadata['Treatment'].sort_values().unique()
    #for treatment in [tl[0]]:
    for treatment in tl:
        filename = output_folder + "interpolations/" + treatment.replace('/', "_") + ".png"
        cprint(f"doing: {filename}", logfile)
        plot_control_cell_to_target_cell(treatment, images, metadata_latent, vae, device, file=filename,  control='DMSO_0.0', control_text = 'DMSO_0.0',  target_text=treatment)

    #### PLOT LATENT SPACE HEATMAP ####
    cprint("Plotting latent space heatmap", logfile)
    # heatmap of (abs) correlations between latent variables and MOA classes
    heatmap_res = heatmap(metadata_latent)
    # sorting latent variables by (abs) sum
    sorted_columns = heatmap_res.sum(axis=0).sort_values(ascending=False).index
    heatmap_res = heatmap_res.reindex(sorted_columns , axis=1)
    # plot heatmap
    plt.figure(figsize = (8,4))
    heat = sns.heatmap(heatmap_res, vmin=0, vmax=0.3)#.set(xticklabels=[])
    figure = heat.get_figure()
    plt.gcf()
    figure.savefig(output_folder + "images/latent_var_heatmap.png", bbox_inches = 'tight')
    plt.close()

    #### NEAREST NEIGHBOR CLASSIFICATION (Not-Same-Compound) ####
    cprint("Nearest neighbor classification (Not-Same-Compound)", logfile)
    targets, predictions = NSC_NearestNeighbor_Classifier(metadata_latent, p=2)

    #### PLOT CONFUSION MATRIX ####
    confusion_matrix = moa_confusion_matrix(targets, predictions)
    cprint(confusion_matrix, logfile)
    df_cm = pd.DataFrame(confusion_matrix/np.sum(confusion_matrix) *100, index = [i for i in mapping],
                            columns = [i for i in mapping])
    plt.figure(figsize = (12,7))
    cm = sns.heatmap(df_cm, annot=True)
    figure = cm.get_figure()
    plt.gcf()
    figure.savefig(output_folder + "images/conf_matrix.png", bbox_inches = 'tight')


    #### PRINT ACCURACY ####
    prec = precision(confusion_matrix)
    acc = Accuracy(confusion_matrix)
    rec = recall(confusion_matrix)
    cprint("Model Accuracy: {}".format(acc), logfile)
    cprint("Model Precision: {}".format(prec), logfile) 
    cprint("Model Recall: {}".format(rec), logfile)
    #cprint("Model F1: {}".format(2 * (precision(confusion_matrix) * recall(confusion_matrix))), logfile)
    cprint("Model F1: {}".format(2 * (prec * rec) / (prec+ rec)), logfile)
