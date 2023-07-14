"""Stats module for gradec."""
import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
from netneurotools.stats import gen_spinsamples
from neuromaps.datasets import fetch_atlas
from nimare.stats import pearson
from nimare.utils import tqdm_joblib
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


def _gen_spinsamples(neuromaps_dir=None, n_samples=1, n_cores=1):
    atlas = fetch_atlas("fsLR", "32k", data_dir=neuromaps_dir, verbose=0)

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

    with tqdm_joblib(tqdm(total=n_samples)):
        results = Parallel(n_jobs=n_cores)(
            delayed(_gen_nullsamples)(coords, hemi, seed) for seed in range(n_samples)
        )

    return np.hstack(results)


def _permtest_pearson(grad_map, metamaps, spinsamples):
    """Permutation test for Pearson correlation."""
    # Calculate true correlations
    n_perm = spinsamples.shape[1]
    corrs = pearson(grad_map, metamaps)

    # Calculate null correlations
    grad_map_null = grad_map[spinsamples]
    corrs_null = [pearson(grad_map_null[:, p_i], metamaps) for p_i in range(n_perm)]

    # Calculate p-values
    n_extreme_corrs = np.sum(np.abs(corrs_null) >= np.abs(corrs), axis=0)
    pvals = n_extreme_corrs / (n_perm + 1)
    corr_pvals = fdr(pvals)

    return corrs, pvals, corr_pvals
