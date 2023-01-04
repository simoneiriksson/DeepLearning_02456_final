import torch
import numpy as np
import pandas as pd
from typing import List, Set, Dict, Tuple, Optional, Any
from collections import defaultdict

from utils.data_preparation import *

def extract_batch_from_indices(indices: np.ndarray, images: torch.Tensor, metadata: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_images = extract_images_from_metadata_indices(indices, images)
    batch_labels = extract_MOA_ids_from_indices(indices, metadata)
    return batch_images, batch_labels

def get_treatment_indices(metadata: pd.DataFrame) -> Dict[str, list]:
    treatments = np.array(metadata[['Image_Metadata_Compound','Image_Metadata_Concentration']])
    result = defaultdict(list)
    
    for i in range(treatments.shape[0]):
        compound, concetration = treatments[i]
        result[(compound, concetration)] += [i]
    
    return result
    
def extract_images_from_metadata_indices(indices: np.ndarray, images: torch.Tensor) -> torch.Tensor:
    return images[indices]

def extract_MOA_ids_from_indices(indices: np.ndarray, metadata: pd.DataFrame) -> torch.Tensor:
    moa_to_id = get_MOA_to_id()
    rows = metadata["moa"].loc[indices]
    return torch.tensor([moa_to_id[row] for row in rows])
