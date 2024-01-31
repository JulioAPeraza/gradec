"""Transforms for gradec."""

import numpy as np
from neuromaps import transforms
from nilearn import image

from gradec.utils import _rm_medial_wall, _zero_medial_wall

# Number of vertices in total without the medial wall
N_VERTICES = {
    "fsLR": {
        "32k": 59412,
        "164k": 298261,
    },
    "fsaverage": {
        "3k": 4661,
        "10k": 18715,
        "41k": 74947,
        "164k": 299881,
    },
    "civet": {
        "41k": 76910,
    },
}


def _vol_to_surf(metamaps, space="fsLR", density="32k", neuromaps_dir=None):
    """Transform 4D metamaps from volume to surface space."""
    n_maps = metamaps.shape[3]

    # TODO: need to parallelize this loop
    metamap_surf = np.zeros((n_maps, N_VERTICES[space][density]))
    for map_i in range(n_maps):
        metamap = image.index_img(metamaps, map_i)

        if space == "fsLR":
            metamap_lh, metamap_rh = transforms.mni152_to_fslr(metamap, fslr_density=density)
        elif space == "fsaverage":
            metamap_lh, metamap_rh = transforms.mni152_to_fsaverage(metamap, fsavg_density=density)
        elif space == "civet":
            metamap_lh, metamap_rh = transforms.mni152_to_civet(metamap, civet_density=density)

        metamap_lh, metamap_rh = _zero_medial_wall(
            metamap_lh,
            metamap_rh,
            space=space,
            density=density,
            neuromaps_dir=neuromaps_dir,
        )
        metamap_arr_lh = metamap_lh.agg_data()
        metamap_arr_rh = metamap_rh.agg_data()

        metamap_surf[map_i, :] = _rm_medial_wall(
            metamap_arr_lh,
            metamap_arr_rh,
            space=space,
            density=density,
            neuromaps_dir=neuromaps_dir,
        )

    return metamap_surf
