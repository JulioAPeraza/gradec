"""Tests for decode."""

import numpy as np

from gradec.decode import GCLDADecoder, LDADecoder, TermDecoder
from gradec.segmentation import KMeansSegmentation


def test_term_corr_decoder(tmp_path_factory, reduced_testdata_laird):
    """Test fetching features from OSF."""
    tmpdir = tmp_path_factory.mktemp("test_fetch_features")

    n_segments = 2
    gradient = np.random.rand(59412)
    segmentation = KMeansSegmentation(n_segments).fit(gradient)
    grad_maps = segmentation.transform()

    decode = TermDecoder(
        n_samples=2,
        feature_group="Neurosynth_TFIDF",
        use_fetchers=False,
        data_dir=tmpdir,
    )
    decode.fit("neurosynth", reduced_testdata_laird)
    corrs_df, pvals_df, corr_pvals_df = decode.transform(grad_maps)

    assert corrs_df.shape == (14, n_segments)
    assert pvals_df.shape == (14, n_segments)
    assert corr_pvals_df.shape == (14, n_segments)


def test_lda_corr_decoder(tmp_path_factory, testdata_laird):
    """Test fetching features from OSF."""
    tmpdir = tmp_path_factory.mktemp("test_fetch_features")

    n_segments = 2
    gradient = np.random.rand(59412)
    segmentation = KMeansSegmentation(n_segments).fit(gradient)
    grad_maps = segmentation.transform()

    n_topics = 10
    decode = LDADecoder(
        n_topics=n_topics,
        n_top_words=3,
        n_samples=2,
        feature_group="Neurosynth_TFIDF",
        use_fetchers=False,
        data_dir=tmpdir,
    )
    decode.fit("neurosynth", testdata_laird)
    corrs_df, pvals_df, corr_pvals_df = decode.transform(grad_maps)

    # Sometimes the number mpas is less than the number of topics,
    # we only look at the number of segments
    assert corrs_df.shape[1] == n_segments
    assert pvals_df.shape[1] == n_segments
    assert corr_pvals_df.shape[1] == n_segments


def test_gclda_corr_decoder(tmp_path_factory, testdata_laird):
    """Test fetching features from OSF."""
    tmpdir = tmp_path_factory.mktemp("test_fetch_features")

    n_segments = 2
    gradient = np.random.rand(59412)
    segmentation = KMeansSegmentation(n_segments).fit(gradient)
    grad_maps = segmentation.transform()

    n_topics = 10
    decode = GCLDADecoder(
        n_iters=10,
        n_samples=2,
        n_topics=n_topics,
        n_top_words=3,
        feature_group="Neurosynth_TFIDF",
        use_fetchers=False,
        data_dir=tmpdir,
    )
    decode.fit("neurosynth", testdata_laird)
    corrs_df, pvals_df, corr_pvals_df = decode.transform(grad_maps, method="correlation")

    assert corrs_df.shape == (n_topics, n_segments)
    assert pvals_df.shape == (n_topics, n_segments)
    assert corr_pvals_df.shape == (n_topics, n_segments)


def test_gclda_decoder(tmp_path_factory, testdata_laird):
    """Test fetching features from OSF."""
    tmpdir = tmp_path_factory.mktemp("test_fetch_features")

    n_segments = 2
    gradient = np.random.rand(59412)
    segmentation = KMeansSegmentation(n_segments).fit(gradient)
    grad_maps = segmentation.transform()

    n_topics = 10
    decode = GCLDADecoder(
        n_iters=10,
        n_topics=n_topics,
        n_top_words=3,
        feature_group="Neurosynth_TFIDF",
        use_fetchers=False,
        data_dir=tmpdir,
    )
    decode.fit("neurosynth", testdata_laird)
    weight_df = decode.transform(grad_maps)

    assert weight_df.shape == (728, n_segments)


def test_term_corr_decoder_fetcher(tmp_path_factory):
    """Test fetching features from OSF."""
    tmpdir = tmp_path_factory.mktemp("test_fetch_features")

    n_segments = 2
    gradient = np.random.rand(59412)
    segmentation = KMeansSegmentation(n_segments).fit(gradient)
    grad_maps = segmentation.transform()

    decode = TermDecoder(n_samples=2, data_dir=tmpdir)
    decode.fit("neurosynth")
    corrs_df, pvals_df, corr_pvals_df = decode.transform(grad_maps)

    assert corrs_df.shape == (3228, n_segments)
    assert pvals_df.shape == (3228, n_segments)
    assert corr_pvals_df.shape == (3228, n_segments)


def test_lda_corr_decoder_fetcher(tmp_path_factory):
    """Test fetching features from OSF."""
    tmpdir = tmp_path_factory.mktemp("test_fetch_features")

    n_segments = 2
    gradient = np.random.rand(59412)
    segmentation = KMeansSegmentation(n_segments).fit(gradient)
    grad_maps = segmentation.transform()

    decode = LDADecoder(
        n_top_words=3,
        n_samples=2,
        data_dir=tmpdir,
    )
    decode.fit("neurosynth")
    corrs_df, pvals_df, corr_pvals_df = decode.transform(grad_maps)

    assert corrs_df.shape == (200, n_segments)
    assert pvals_df.shape == (200, n_segments)
    assert corr_pvals_df.shape == (200, n_segments)


def test_gclda_corr_decoder_fetcher(tmp_path_factory, testdata_laird):
    """Test fetching features from OSF."""
    tmpdir = tmp_path_factory.mktemp("test_fetch_features")

    n_segments = 2
    gradient = np.random.rand(59412)
    segmentation = KMeansSegmentation(n_segments).fit(gradient)
    grad_maps = segmentation.transform()

    decode = GCLDADecoder(n_samples=2, n_top_words=3, data_dir=tmpdir)
    decode.fit("neurosynth")
    corrs_df, pvals_df, corr_pvals_df = decode.transform(grad_maps, method="correlation")

    assert corrs_df.shape == (200, n_segments)
    assert pvals_df.shape == (200, n_segments)
    assert corr_pvals_df.shape == (200, n_segments)


def test_gclda_decoder_fetcher(tmp_path_factory, testdata_laird):
    """Test fetching features from OSF."""
    tmpdir = tmp_path_factory.mktemp("test_fetch_features")

    n_segments = 2
    gradient = np.random.rand(59412)
    segmentation = KMeansSegmentation(n_segments).fit(gradient)
    grad_maps = segmentation.transform()

    decode = GCLDADecoder(
        n_top_words=3,
        data_dir=tmpdir,
    )
    decode.fit("neurosynth", testdata_laird)
    weight_df = decode.transform(grad_maps)

    assert weight_df.shape == (3228, n_segments)
