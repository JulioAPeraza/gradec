"""Miscellaneous functions used for analyses."""
import logging
import multiprocessing as mp
import os

import nibabel as nib
import numpy as np
from neuromaps.datasets import fetch_atlas
from nibabel.gifti import GiftiDataArray
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from sklearn.metrics import pairwise_distances

LGR = logging.getLogger(__name__)


def _reorder_matrix(mat, reorder):
    """Reorder a matrix.

    This function reorders the provided matrix. It was adaptes from
    nilearn.plotting.plot_matrix._reorder_matrix.

        License
    -------
    New BSD License
    Copyright (c) 2007 - 2023 The nilearn developers.
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    a. Redistributions of source code must retain the above copyright notice,
        this list of conditions and the following disclaimer.
    b. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
    c. Neither the name of the nilearn developers nor the names of
        its contributors may be used to endorse or promote products
        derived from this software without specific prior written
        permission.
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
    OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
    DAMAGE.
    """
    reorder = "average" if reorder is True else reorder

    linkage_matrix = linkage(mat, method=reorder)
    ordered_linkage = optimal_leaf_ordering(linkage_matrix, mat)

    return leaves_list(ordered_linkage)


def get_data_dir(data_dir=None):
    """
    Gets path to gradec data directory

    Parameters
    ----------
    data_dir : str, optional
        Path to use as data directory. If not specified, will check for
        environmental variable 'GRADEC_DATA'; if that is not set, will
        use `~/gradec-data` instead. Default: None

    Returns
    -------
    data_dir : str
        Path to use as data directory

     Notes
    -----
    Taken from Neuromaps.
    https://github.com/netneurolab/neuromaps/blob/abf5a5c3d3d011d644b56ea5c6a3953cedd80b37/
    neuromaps/datasets/utils.py#LL91C1-L115C20
    """

    if data_dir is None:
        data_dir = os.environ.get("GRADEC_DATA", os.path.join("~", "gradec-data"))
    data_dir = os.path.expanduser(data_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    return data_dir


def rm_fslr_medial_wall(data_lh, data_rh, neuromaps_dir, join=True):
    """Remove medial wall from data in fsLR space
    Data in 32k fs_LR space (e.g., Human Connectome Project data) often in
    GIFTI format include the medial wall in their data arrays, which results
    in a total of 64984 vertices across hemispheres. This function removes
    the medial wall vertices to produce a data array with the full 59412 vertices,
    which is used to perform functional decoding.

    This function was adapted from :func:`surfplot.utils.add_fslr_medial_wall`.

    Parameters
    ----------
    data : numpy.ndarray
        Surface vertices. Must have exactly 32492 vertices per hemisphere.
    join : bool
        Return left and right hemipsheres in the same arrays. Default: True
    Returns
    -------
    numpy.ndarray
        Vertices with medial wall excluded (59412 vertices total)
    Raises
    ------
    ValueError
        `data` has the incorrect number of vertices (59412 or 64984 only
        accepted)
    """
    assert data_lh.shape[0] == 32492
    assert data_rh.shape[0] == 32492

    atlas = fetch_atlas("fsLR", "32k", data_dir=neuromaps_dir, verbose=0)
    medial_lh, medial_rh = atlas["medial"]
    wall_lh = nib.load(medial_lh).agg_data()
    wall_rh = nib.load(medial_rh).agg_data()

    data_lh = data_lh[np.where(wall_lh != 0)]
    data_rh = data_rh[np.where(wall_rh != 0)]

    if not join:
        return data_lh, data_rh

    data = np.hstack((data_lh, data_rh))
    assert data.shape[0] == 59412
    return data


def zero_fslr_medial_wall(data_lh, data_rh, neuromaps_dir):
    """Remove medial wall from data in fsLR space"""

    atlas = fetch_atlas("fsLR", "32k", data_dir=neuromaps_dir, verbose=0)
    medial_lh, medial_rh = atlas["medial"]
    medial_arr_lh = nib.load(medial_lh).agg_data()
    medial_arr_rh = nib.load(medial_rh).agg_data()

    data_arr_lh = data_lh.agg_data()
    data_arr_rh = data_lh.agg_data()
    data_arr_lh[np.where(medial_arr_lh == 0)] = 0
    data_arr_rh[np.where(medial_arr_rh == 0)] = 0

    data_lh.remove_gifti_data_array(0)
    data_rh.remove_gifti_data_array(0)
    data_lh.add_gifti_data_array(GiftiDataArray(data_arr_lh))
    data_rh.add_gifti_data_array(GiftiDataArray(data_arr_rh))

    return data_lh, data_rh


def affinity(matrix, sparsity):
    # Generate percentile thresholds for 90th percentile
    perc = np.array([np.percentile(x, sparsity) for x in matrix])

    # Threshold each row of the matrix by setting values below 90th percentile to 0
    for i in range(matrix.shape[0]):
        matrix[i, matrix[i, :] < perc[i]] = 0
    matrix[matrix < 0] = 0

    # Now we are dealing with sparse vectors. Cosine similarity is used as affinity metric
    matrix = 1 - pairwise_distances(matrix, metric="cosine")

    return matrix


def get_resource_path():
    """Return the path to general resources, terminated with separator.

    Resources are kept outside package folder in "datasets".
    Based on function by Yaroslav Halchenko used in Neurosynth Python package.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "resources") + os.path.sep)


def _check_ncores(n_cores):
    """Check number of cores used for method. Taken from nimare.utils."""
    if n_cores <= 0:
        n_cores = mp.cpu_count()
    elif n_cores > mp.cpu_count():
        LGR.warning(
            f"Desired number of cores ({n_cores}) greater than number "
            f"available ({mp.cpu_count()}). Setting to {mp.cpu_count()}."
        )
        n_cores = mp.cpu_count()
    return n_cores
