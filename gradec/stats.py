"""Stats module for gradec."""

import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from netneurotools.stats import gen_spinsamples
from neuromaps.datasets import fetch_atlas
from nimare.stats import pearson
from pymare.stats import fdr
from tqdm.auto import tqdm


def _gen_nullsamples(coords, hemi, seed):
    return gen_spinsamples(
        coords,
        hemi,
        n_rotate=1,
        seed=seed,
        method="vasa",
        check_duplicates=True,
        verbose=True,
    )


def _gen_spinsamples(space="fsLR", density="32k", neuromaps_dir=None, n_samples=1, n_cores=1):
    atlas = fetch_atlas(space, density, data_dir=neuromaps_dir, verbose=0)

    sphere_lh, sphere_rh = atlas["sphere"]
    coords_lh, _ = nib.load(sphere_lh).agg_data()
    coords_rh, _ = nib.load(sphere_rh).agg_data()

    medial_lh, medial_rh = atlas["medial"]
    medial_arr_lh = nib.load(medial_lh).agg_data()
    medial_arr_rh = nib.load(medial_rh).agg_data()

    coords_lh = coords_lh[np.where(medial_arr_lh != 0)]
    coords_rh = coords_rh[np.where(medial_arr_rh != 0)]

    hemi = np.hstack((np.zeros((coords_lh.shape[0],)), np.ones((coords_rh.shape[0],))))
    coords = np.vstack((coords_lh, coords_rh))

    results = [
        r
        for r in tqdm(
            Parallel(return_as="generator", n_jobs=n_cores)(
                delayed(_gen_nullsamples)(coords, hemi, seed) for seed in range(n_samples)
            ),
            total=n_samples,
        )
    ]

    return np.hstack(results)


def _permtest_pearson(grad_map, metamaps, spinsamples=None):
    """Permutation test for Pearson correlation."""
    # Calculate true correlations
    corrs = pearson(grad_map, metamaps)

    if spinsamples is not None:
        # Calculate null correlations
        n_perm = spinsamples.shape[1]
        grad_map_null = grad_map[spinsamples]
        corrs_null = [pearson(grad_map_null[:, p_i], metamaps) for p_i in range(n_perm)]

        # Calculate p-values
        n_extreme_corrs = np.sum(np.abs(corrs_null) >= np.abs(corrs), axis=0)
        pvals = n_extreme_corrs / (n_perm + 1)
        corr_pvals = fdr(pvals)

    else:
        pvals, corr_pvals = pd.DataFrame(), pd.DataFrame()

    return corrs, pvals, corr_pvals
