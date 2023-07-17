"""Transforms for gradec."""
import numpy as np
from neuromaps import transforms
from nilearn import image

from gradec.utils import _rm_fslr_medial_wall, _zero_fslr_medial_wall

N_VERTICES = {
    "fsLR": {
        "32k": 59412,
        "164k": 298261,
    }
}


def _mni152_to_fslr(metamaps, fslr_density="32k", neuromaps_dir=None):
    """Transform 4D metamaps from MNI152 to fsLR space."""
    n_maps = metamaps.shape[3]

    # TODO: need to parallelize this loop
    metamap_fslr = np.zeros((n_maps, N_VERTICES["fsLR"][fslr_density]))
    for map_i in range(n_maps):
        metamap = image.index_img(metamaps, map_i)

        metamap_lh, metamap_rh = transforms.mni152_to_fslr(metamap, fslr_density=fslr_density)
        metamap_lh, metamap_rh = _zero_fslr_medial_wall(metamap_lh, metamap_rh, neuromaps_dir)
        metamap_arr_lh = metamap_lh.agg_data()
        metamap_arr_rh = metamap_rh.agg_data()
        metamap_fslr[map_i, :] = _rm_fslr_medial_wall(
            metamap_arr_lh, metamap_arr_rh, neuromaps_dir
        )

    return metamap_fslr
