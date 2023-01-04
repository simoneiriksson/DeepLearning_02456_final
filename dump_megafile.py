
from utils.utils import cprint, get_datetime, create_logfile, constant_seed, StatusString
from utils.data_preparation import create_directory, read_metadata, get_relative_image_paths, load_images
from utils.data_preparation import read_metadata_and_images, get_server_directory_path
import torch
import math

output_folder = "./dump/dump_megafile/"
create_directory(output_folder)
constant_seed()

logfile = create_logfile(output_folder + "downstream_log.log")
cprint("output_folder is: {}".format(output_folder), logfile)

path = get_server_directory_path()
#path = '../data/all/'
create_directory('../data/')
metadata_all = read_metadata(path + "metadata.csv") 
cprint("loading images from individual files", logfile)
relative_paths = get_relative_image_paths(metadata_all)
image_paths = [path + relative for relative in relative_paths] #absolute path
log_every = max(math.floor(len(image_paths)/100),10)
cprint("loading images now", logfile)
images = load_images(image_paths, verbose=True, log_every=log_every, logfile=logfile)
torch.save(images, "../data/images.pt")
