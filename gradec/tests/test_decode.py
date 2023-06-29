"""Tests for decode."""
import numpy as np
import pytest

from gradec.decode import CorrelationDecoder, GCLDADecoder
from gradec.segmentation import KMeansSegmentation


@pytest.mark.skip(reason="This is too slow for GitHub action at the moment.")
@pytest.mark.parametrize(
    "dset_nm,model_nm",
    [
        ("neurosynth", "term"),
        ("neurosynth", "lda"),
        ("neuroquery", "term"),
        ("neuroquery", "lda"),
    ],
)
def test_correlation_decoder(tmp_path_factory, dset_nm, model_nm):
    """Test fetching features from OSF."""
    tmpdir = tmp_path_factory.mktemp("test_fetch_features")

    n_segments = 2
    gradient = np.random.rand(59412)
    segmentation = KMeansSegmentation(n_segments).fit(gradient)
    grad_maps = segmentation.transform()

    decode = CorrelationDecoder(model_nm=model_nm, data_dir=tmpdir)
    decode.fit(dset_nm)
    corrs_df, pvals_df, corr_pvals_df = decode.transform(grad_maps)

    assert corrs_df.shape == (200, n_segments)
    assert pvals_df.shape == (200, n_segments)
    assert corr_pvals_df.shape == (200, n_segments)


@pytest.mark.parametrize(
    "dset_nm,model_nm,n_maps",
    [
        ("neurosynth", "gclda", 3228),
        ("neuroquery", "gclda", 6145),
    ],
)
def test_gclda_decoder(tmp_path_factory, dset_nm, model_nm, n_maps):
    """Test fetching features from OSF."""
    tmpdir = tmp_path_factory.mktemp("test_fetch_features")

    n_segments = 5
    gradient = np.random.rand(59412)
    segmentation = KMeansSegmentation(n_segments).fit(gradient)
    grad_maps = segmentation.transform()

    decode = GCLDADecoder(model_nm=model_nm, data_dir=tmpdir)
    decode.fit(dset_nm)
    decoded_df = decode.transform(grad_maps)

    assert decoded_df.shape == (n_maps, n_segments)
