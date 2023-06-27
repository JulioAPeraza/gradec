"""Stats module for gradec."""
import numpy as np
from nimare.stats import pearson
from pymare.stats import fdr

N_VERTICES = 59412
N_PERMUTATIONS = 1000


def _permute_metamaps(metamaps):
    """Permute metamaps."""
    # n_metamaps = metamaps.shape[0]
    # meta_maps_permuted_arr = np.zeros((n_metamaps, N_VERTICES, N_PERMUTATIONS))
    # meta_maps_permuted_arr[metamap_i, :, :] = meta_map_fslr[nullsamples]
    pass


def _permtest_pearson(grad_map, metamaps, spinsamples):
    """Permutation test for Pearson correlation."""
    # Calculate true correlations
    n_perm = spinsamples.shape[1]
    corrs = pearson(grad_map, metamaps)

    # Calculate null correlations
    grad_map_null = grad_map[spinsamples]
    corrs_null = [pearson(grad_map_null[:, p_i], metamaps) for p_i in range(n_perm)]

    # Calculate p-values
    n_extreme_corrs = np.sum(np.abs(corrs) >= np.abs(corrs_null), axis=0)
    pvals = n_extreme_corrs / n_perm
    corr_pvals = fdr(pvals)

    return corrs, pvals, corr_pvals
