"""Test fetching data from OSF."""
import numpy as np
import pytest

from gradec.fetcher import (
    _fetch_features,
    _fetch_metamaps,
    _fetch_neuroquery_counts,
    _fetch_spinsamples,
)


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
def test_fetch_features(tmp_path_factory, dset_nm, model_nm):
    """Test fetching features from OSF."""
    tmpdir = tmp_path_factory.mktemp("test_fetch_features")

    features = _fetch_features(dset_nm, model_nm, data_dir=tmpdir)
    assert isinstance(features, list)
    assert len(features) > 0
    assert isinstance(features[0], list)
    assert len(features[0]) > 0


@pytest.mark.parametrize(
    "dset_nm,model_nm,n_maps",
    [
        ("neurosynth", "term", 3228),
        ("neurosynth", "lda", 200),
        ("neurosynth", "gclda", 200),
        ("neuroquery", "term", 6145),
        ("neuroquery", "lda", 200),
        ("neuroquery", "gclda", 200),
    ],
)
def test_fetch_metamaps(tmp_path_factory, dset_nm, model_nm, n_maps):
    """Test fetching features from OSF."""
    tmpdir = tmp_path_factory.mktemp("test_fetch_features")

    metamaps_fslr = _fetch_metamaps(dset_nm, model_nm, data_dir=tmpdir)
    assert isinstance(metamaps_fslr, np.ndarray)
    assert metamaps_fslr.shape == (n_maps, 59412)


def test_fetch_spinsamples(tmp_path_factory):
    """Test fetching spinsamples from OSF."""
    tmpdir = tmp_path_factory.mktemp("test_fetch_spinsamples")

    spinsamples = _fetch_spinsamples(data_dir=tmpdir)
    assert isinstance(spinsamples, np.ndarray)
    assert spinsamples.shape == (59412, 1000)


def test_fetch_neuroquery_counts(tmp_path_factory):
    """Test fetching neuroquery_counts from OSF."""
    tmpdir = tmp_path_factory.mktemp("test_fetch_features")

    counts = _fetch_neuroquery_counts(data_dir=tmpdir)
    assert isinstance(counts, np.ndarray)
