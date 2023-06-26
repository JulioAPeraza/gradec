"""Transforms for gradec."""
import numpy as np
import sparse
from neuromaps import transforms
from nilearn import image

from gradec.utils import rm_fslr_medial_wall, zero_fslr_medial_wall

N_VERTICES = 59412
N_PERMUTATIONS = 1000


def _mni152_to_fslr(metamaps, neuromaps_dir=None):
    """Transform 4D metamaps from MNI152 to fsLR space."""
    n_maps = metamaps.shape[3]

    metamap_fslr = np.zeros((n_maps, N_VERTICES))
    for map_i in range(n_maps):
        metamap = image.index_img(metamaps, map_i)

        metamap_lh, metamap_rh = transforms.mni152_to_fslr(metamap)
        metamap_lh, metamap_rh = zero_fslr_medial_wall(metamap_lh, metamap_rh, neuromaps_dir)
        metamap_arr_lh = metamap_lh.agg_data()
        metamap_arr_rh = metamap_rh.agg_data()
        metamap_fslr[map_i, :] = rm_fslr_medial_wall(metamap_arr_lh, metamap_arr_rh, neuromaps_dir)

    return metamap_fslr
