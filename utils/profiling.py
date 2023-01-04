import pandas as pd
import numpy as np
from utils.utils import cprint

# extracting latent variables for each image/cell
def LatentVariableExtraction(metadata, images, batch_size, vae, device, logfile=None):
    metadata['Well_unique'] = metadata['Image_Metadata_Well_DAPI'] + '_' + metadata['Image_Metadata_Plate_DAPI']
    metadata['Treatment'] = metadata['Image_Metadata_Compound'] + '_' + metadata['Image_Metadata_Concentration'].astype(str)
    metadata['week'] = metadata['Image_PathName_DAPI'].str.split("_", n=1, expand = True)[0]
    metadata['row_id'] = np.arange(len(metadata))
    batch_size=min(batch_size, len(images))
    batch_offset = np.arange(start=0, stop=images.shape[0]+1, step=batch_size)

    df = pd.DataFrame()
    new_metadata = pd.DataFrame()

    for j, item in enumerate(batch_offset[:-1]):
        start = batch_offset[j]
        end = batch_offset[j+1]
        image_subset = images[start:end,:,:,:]
        outputs = vae(image_subset.to(device))
        z = outputs["z"].to('cpu')
        columns_list = ["latent_"+str(z) for z in range(z.shape[1])]
        z_df = pd.DataFrame(z.detach().numpy(), columns=columns_list)
        z_df.index = list(range(start,end))
        df = pd.concat([metadata.iloc[start:end], z_df], axis=1)
        new_metadata = pd.concat([new_metadata, df], axis=0)
        #cprint("Profiling {}/{} batches of size {}".format(j, len(batch_offset)-1, batch_size), logfile)
        cprint(f"Profiling {j}/{len(batch_offset)-1} batches of size {batch_size}", logfile)
    
    # last batch
    start = batch_offset[-1]
    end = images.shape[0]
    if start != end:
        #print(start, end)
        image_subset = images[start:end,:,:,:]
        outputs = vae(image_subset.to(device))
        z = outputs["z"].to('cpu')
        #print("z.shape", z.shape)
        columns_list = ["latent_"+str(z) for z in range(z.shape[1])]
        z_df = pd.DataFrame(z.detach().numpy(), columns=columns_list)
        z_df.index = list(range(start,end))
        df = pd.concat([metadata.iloc[start:end], z_df], axis=1)
        new_metadata = pd.concat([new_metadata, df], axis=0)
        cprint("Profiling {}/{} batches of size {}".format(len(batch_offset)-1, len(batch_offset)-1, end-start), logfile)

    return new_metadata

  # Wells Profiles
def well_profiles(nm):
    latent_cols = [col for col in nm.columns if type(col)==str and col[0:7]=='latent_']
    wa = nm.groupby('Image_Metadata_Well_DAPI').mean()[latent_cols]
    return wa

# function to get the cell closest to each Well profile
def well_center_cells(df,well_profiles,p=2):
    wcc = []
    latent_cols = [col for col in df.columns if type(col)==str and col[0:7]=='latent_']
    for w in well_profiles.index:
        diffs = (abs(df[df['Image_Metadata_Well_DAPI'] == w][latent_cols] - well_profiles.loc[w])**p)
        diffs_sum = diffs.sum(axis=1)**(1/p)
        diffs_min = diffs_sum.min()
        wcc.append(diffs[diffs_sum == diffs_min].index[0])
    return df.loc[wcc]

# Treatment Profiles
def treatment_profiles(nm):
    latent_cols = [col for col in nm.columns if type(col)==str and col[0:7]=='latent_']
    mean_over_treatment_well_unique = nm.groupby(['Treatment', 'Image_Metadata_Compound', 'Image_Metadata_Concentration','Well_unique', 'moa'], as_index=False).mean()
    median_over_treatment = mean_over_treatment_well_unique.groupby(['Treatment', 'Image_Metadata_Compound', 'Image_Metadata_Concentration', 'moa'], as_index=False).median()
    return median_over_treatment
    
# Treatment Profiles
#def treatment_profiles(df):
#  t = df.groupby(['Treatment','Well_unique'], as_index=False).mean().groupby(['Treatment']).median().iloc[:,-256:]
#  return t

# function to get the cell closest to each Treatment profile

#def treatment_center_cells(df,treatment_profiles,p=2):
#  tcc = []
#  for t in treatment_profiles.index:
#    diffs = (abs(df[df['Treatment'] == t].iloc[:,-256:] - treatment_profiles.loc[t])**p)
#    diffs_sum = diffs.sum(axis=1)**(1/p)
#    diffs_min = diffs_sum.min()
#    tcc.append(diffs[diffs_sum == diffs_min].index[0])
#  return df.loc[tcc]

# function to get the cell closest to each Treatment profile
def treatment_center_cells(df,treatment_profiles,p=2):
    tcc = []
    latent_cols = [col for col in df.columns if type(col)==str and col[0:7]=='latent_']
    treatment_profiles = treatment_profiles.set_index('Treatment')
    for t in treatment_profiles.index:
        diffs = (abs(df[df['Treatment'] == t][latent_cols] - treatment_profiles.loc[t])**p)
        diffs_sum = diffs.sum(axis=1)**(1/p)
        diffs_min = diffs_sum.min()
        tcc.append(diffs[diffs_sum == diffs_min].index[0])
    #tcc = df.loc[tcc]    
    #tcc = tcc.set_index('Treatment')
    return df.loc[tcc]


# Compount/Concentration Profiles
def cc_profile(nm):
    latent_cols = [col for col in df.columns if type(col)==str and col[0:7]=='latent_']
    cc =  nm.groupby(['Image_Metadata_Compound','Image_Metadata_Concentration']).median()[latent_cols]
    return cc

# function to get the cell closest to each Compound/Concentration profile
def cc_center_cells(df,cc_profiles,p=2):
    cc_center_cells = []
    latent_cols = [col for col in df.columns if type(col)==str and col[0:7]=='latent_']
    for cc in cc_profiles.index:
        diffs = (abs(df[(df['Image_Metadata_Compound'] == cc[0]) & (df['Image_Metadata_Concentration'] == cc[1])][latent_cols] - cc_profiles.loc[cc]))**p
        diffs_sum = diffs.sum(axis=1)**(1/p)
        diffs_min = diffs_sum.min()
        cc_center_cells.append(diffs[diffs_sum == diffs_min].index[0])
    return df.loc[cc_center_cells]

def NSC_NearestNeighbor_Classifier(metadata_latent, p=2):
    treatment_profiles_df = treatment_profiles(metadata_latent)
    latent_cols = [col for col in metadata_latent.columns if type(col)==str and col[0:7]=='latent_']
    
    for compound in metadata_latent['Image_Metadata_Compound'].unique():
        A_set = treatment_profiles_df[treatment_profiles_df['Image_Metadata_Compound'] == compound]
        B_set = treatment_profiles_df[treatment_profiles_df['Image_Metadata_Compound'] != compound]
        for A in A_set.index:
            A_treatment = A_set.loc[A]['Treatment']
            diffs = (abs(B_set[latent_cols] - A_set.loc[A][latent_cols]))**p
            diffs_sum = diffs.sum(axis=1)**(1/p)
            diffs_min = diffs_sum.min()
            treatment_profiles_df.loc[treatment_profiles_df['Treatment']==A_treatment,'moa_pred'] = B_set.at[diffs[diffs_sum == diffs_min].index[0], 'moa']
    return treatment_profiles_df['moa'], treatment_profiles_df['moa_pred']

    
def moa_confusion_matrix(targets, predictions):
    nb_classes = len(targets.unique())
    moa_classes = targets.sort_values().unique()
    classes = np.zeros((nb_classes, nb_classes))
    for i in range(nb_classes):
        for j in range(nb_classes):
            for t in range(len(targets)):
                if targets[t] == moa_classes[i] and predictions[t] == moa_classes[j]:
                    classes[i,j] += 1
    confusion_matrix = classes  
    return confusion_matrix

def Accuracy(confusion_matrix):
    class_accuracy = 100*confusion_matrix.diagonal()/confusion_matrix.sum(1)
    class_accuracy = class_accuracy.mean()
    return class_accuracy

    
def precision(confusion_matrix):
    truepos = np.diag(confusion_matrix)
    precision = truepos / np.sum(confusion_matrix, axis=0)
    return precision.mean() * 100


def recall(confusion_matrix):
    truepos = np.diag(confusion_matrix)
    recall = truepos / np.sum(confusion_matrix, axis=1)
    return recall.mean() * 100
