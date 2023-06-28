import os

import numpy as np
import pytest

from gradec.decode import CorrelationDecoder
from gradec.segmentation import KMeansSegmentation


@pytest.mark.parametrize(
    "dset_nm,model_nm",
    [
        ("neurosynth", "term"),
        ("neurosynth", "lda"),
        ("neurosynth", "gclda"),
        ("neuroquery", "term"),
        ("neuroquery", "lda"),
        ("neuroquery", "gclda"),
    ],
)
def test_correlation_decoder(tmp_path_factory, dset_nm, model_nm):
    """Test fetching features from OSF."""
    # tmpdir = tmp_path_factory.mktemp("test_fetch_features")
    tmpdir = "/Users/jperaza/Desktop/tes_fetcher"
    n_segments = 5
    # gradient = np.random.rand(59412)
    gradient = np.load(
        "/Users/jperaza/Documents/GitHub/gradient-decoding/results/gradient/principal_gradient.npy"
    )
    segmentation = KMeansSegmentation(n_segments).fit(gradient)
    grad_maps = segmentation.transform()

    decode = CorrelationDecoder(model_nm=model_nm, basepath=tmpdir, n_cores=5)
    decode.fit(dset_nm)
    corrs_df, pvals_df, corr_pvals_df = decode.transform(grad_maps, reorder=True)

    corrs_df.to_csv(os.path.join(tmpdir, "corrs_df.csv"))
    pvals_df.to_csv(os.path.join(tmpdir, "pvals_df.csv"))
    corr_pvals_df.to_csv(os.path.join(tmpdir, "corr_pvals_df.csv"))

    assert corrs_df.shape == (200, n_segments)
    assert pvals_df.shape == (200, n_segments)
    assert corr_pvals_df.shape == (200, n_segments)
