import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

import torch
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.BRAIN_ViT_config import h_init, w_init, d_init, device
from data.atlas_scores.high_imp_regions import get_high_regions
# sys.path.pop(0)
# sys.path.pop(0)

def prepare_regions():
    # Load the JHU WM atlas .nii.gz file
    file_path = 'data/atlas_scores/JHU-ICBM-labels-1mm.nii.gz'  # Replace with your file path
    wm_img = nib.load(file_path)
    wm_atlas_data = wm_img.get_fdata()

    # Load the AAL3v1 atlas .nii.gz file
    file_path = 'data/atlas_scores/AAL3v1_1mm.nii.gz'  # Replace with your file path
    img = nib.load(file_path)
    aal_atlas = img.get_fdata()

    # Print or explore the data
    print("AAL atlas:",aal_atlas.shape)  # Prints the shape of the 3D/4D data
    print("JHU WM atlas:",wm_atlas_data.shape)  # Prints the shape of the 3D/4D data

    #zoom atlas
    # Get atlas and data arrays
    atlas_data = aal_atlas.copy()

    # Calculate the zoom factors for resampling the atlas
    zoom_factors = (
        h_init / atlas_data.shape[0],
        w_init / atlas_data.shape[1],
        d_init / atlas_data.shape[2]
    )

    # Resample the atlas data
    aal_atlas_resampled_data = zoom(atlas_data, zoom_factors, order=0)  # Order=0 for nearest-neighbor

    # print("Resampled atlas shape: ",aal_atlas_resampled_data.shape, "parcels (+background): ",len(np.unique(aal_atlas_resampled_data)))
    
    aal_atlas_data_r = np.array(aal_atlas_resampled_data, dtype=int)
    # aal_all_parcels = np.unique(aal_atlas_data_r)
    wm_atlas_data_r = np.array(wm_atlas_data, dtype=int)
    # wm_all_parcels = np.unique(wm_atlas_data_r)

    # Lists of high relevance regions
    AAL3_high_relevance, JHU_WM_high_relevance = get_high_regions()
    wm_high_relevance = set(JHU_WM_high_relevance)
    aal_high_relevance = set(AAL3_high_relevance)

    # Create masks for high relevance and non-zero image regions
    wm_high_relevance_mask = np.isin(wm_atlas_data_r, list(wm_high_relevance))
    aal_high_relevance_mask = np.isin(aal_atlas_data_r, list(aal_high_relevance))
    high_relevance_mask = np.logical_or(wm_high_relevance_mask, aal_high_relevance_mask)

    return torch.tensor(high_relevance_mask, dtype=torch.bool)


def process_batch(batch_data, high_relevance_mask):
    high_relevance_mask = high_relevance_mask.to(device)
    # Reshape high_relevance_mask to match batch dimensions for broadcasting
    high_relevance_mask_expanded = high_relevance_mask.unsqueeze(0).expand(batch_data.shape[0],-1, -1, -1, -1)

    # Create masks for non-zero image regions
    non_zero_smri_mask = batch_data != 0

    # Combine masks to identify low importance regions
    low_relevance_mask = non_zero_smri_mask & ~high_relevance_mask_expanded

    # Initialize atlas data scores matrix
    atlas_data_scores = torch.zeros_like(batch_data, dtype=torch.int)

    # Set high and low relevance values
    atlas_data_scores[high_relevance_mask_expanded] = 2
    atlas_data_scores[low_relevance_mask] = 1

    return atlas_data_scores
