import os

import pytest

from gradec.fetcher import _fetch_features


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
    # tmpdir = tmp_path_factory.mktemp("test_fetch_features")
    tmpdir = "/Users/jperaza/Desktop/tes_fetcher"

    features = _fetch_features(dset_nm, model_nm, data_dir=tmpdir, resume=True, verbose=9)
    assert isinstance(features, list)
    assert len(features) > 0
    assert isinstance(features[0], list)
    assert len(features[0]) > 0
