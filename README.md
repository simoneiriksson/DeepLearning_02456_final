# GENERATIVE MODELLING FOR PHENOTYPIC PROFILING

Authors: Andrea Valletta (S180768), Julia Cathrine Youngman (S123540) and Simon Daniel Eiriksson (S180722)

This repository contains code and some trained models for the final project in the course Deep Learning 02456 at DTU Compute, Fall 2022. We have not submitted a Jupyter Notebook since the necesary compute ressources to run this code are only available at compute clusters.

## Prerequisites
The code has been run using the following modules/versions:

* python3/3.8.2
* pandas/1.0.3
* matplotlib/3.2.1
* cuda/11.5
* cudnn/v8.3.2.44-prod-cuda-11.5
* seaborn 0.12.1
* numpy 1.18.2
* torch 1.13.0

## Setup
In order to train a new model, run the script jobscript.sh with the command `bsub < jobscript.sh`. In this script the HPC parameters are specified. Specifications of what model one wish to train must be entered into script_gpu_cluster.py, into the params dictionary.

As it is set up now, the program trains the SparseVAEGAN model, with 50 epochs, alpha=.1 and beta=1, using all the images in the dataset. The duration of the training last for around 4-5 hours on the gpuv100 queue. If one wants to reduce the amount of files used in training the model, one can change the load_subset_of_images parameter in the read_metadata_and_images function to something smaller, such as load_subset_of_images = 10000.

## Files and folders
The repository has the following overall structure:
* /
    * /jobscript.sh: Script that submits the file script_gpu_cluster.py to the HPC cluster.
    * /script_gpu_cluster.py: Main script that trains a model on single cell image files and run the down stream classification tasks and image plotting etc.
    * /downstream_jobscript.sh: Script that submits the file downstream_only.py to the HPC cluster
    * /downstream_only.py: Script that only runs the down stream classification tasks, given a model that is already trained. 
    * /downstream_task.py: Function definition containing the downstream classification task
    * /VAE_trainer.py: Contains the training loop for the CytoVAE and SparseVAE models.
    * /VAEGAN_trainer.py: Contains the training loop for the CytoVAEGAN and SparseVAEGAN models.
    * /dump_megafile.py: Contains code that copies all the image files into one images.npy file. For speed-up purpose, since it takes around 40 minutes to load all the 480.000 individual image files.
    * /README.md: This file


* models/
    * models/VariationalInference_SparseVAEGAN.py: Inference class for the SparseVAEGAN model.
    * models/VariationalInference_VAE.py: Inference class for the CytoVAE model.
    * models/VariationalInference_VAEGAN.py: Inference class for the VAEGAN model.
    * models/CytoVariationalAutoencoder.py: Model definition for the CytoVAE model.
    * models/SparseVariationalAutoencoder.py: Model definition for the SparseVAE model.
    * models/VariationalInference_SparseVAE.py: Inference class for the SparseVAE model.
    * models/DISC.py: Model definition for the discriminator for the CytoVAEGAN and SparseVAEGAN models.
    * models/LoadModels.py: Small utility to initiate new models and load pretrained models.
    * models/PrintSize.py: Small utility to print layer size in Sequential models. For debugging purposes.
    * models/ReparameterizedDiagonalGaussian.py: Not used.

* utils/ Various auxilliary .py files.

* dump/ where trained models are saved along with output images.
