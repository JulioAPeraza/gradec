"""Tests for gradec.plot module."""

import matplotlib as mpl
import numpy as np

from gradec.plot import plot_cloud, plot_radar, plot_surf_maps


def test_plot_radar(tmp_path_factory):
    """Test plotting radar chart."""
    tmpdir = tmp_path_factory.mktemp("test_radarplot.png")

    corrs = [0.8, 0.6, 0.4, 0.2]
    features = ["Feature 1", "Feature 2", "Feature 3", "Feature 4"]
    model_nm = "term"

    fig = plot_radar(corrs, features, model_nm)
    assert isinstance(fig, mpl.figure.Figure)

    # Test with out_fig
    out_fig = tmpdir / "test_surfplot.png"

    plot_radar(corrs, features, model_nm, out_fig=out_fig)
    assert out_fig.exists()

    # Test with lda-based models
    corrs = [0.8, 0.6, 0.4]
    features = [
        ["Feature-1.1", "Feature-1.2", "Feature-1.3"],
        ["Feature-2.1", "Feature-2.2", "Feature-2.3"],
        ["Feature-3.1", "Feature-3.2", "Feature-3.3"],
    ]
    model_nm = "lda"

    fig = plot_radar(corrs, features, model_nm)
    assert isinstance(fig, mpl.figure.Figure)


def test_plot_cloud(tmp_path_factory):
    """Test plotting word cloud."""
    tmpdir = tmp_path_factory.mktemp("test_wordcloud.png")

    corrs = [0.8, 0.6, 0.4, 0.2]
    features = ["Feature 1", "Feature 2", "Feature 3", "Feature 4"]
    model_nm = "term"

    fig = plot_cloud(corrs, features, model_nm)
    assert isinstance(fig, mpl.figure.Figure)

    # Test with out_fig
    out_fig = tmpdir / "test_surfplot.png"

    plot_cloud(corrs, features, model_nm, out_fig=out_fig)
    assert out_fig.exists()

    # Test with lda-based models
    corrs = [0.8, 0.6, 0.4]
    features = [
        ["Feature-1.1", "Feature-1.2", "Feature-1.3"],
        ["Feature-2.1", "Feature-2.2", "Feature-2.3"],
        ["Feature-3.1", "Feature-3.2", "Feature-3.3"],
    ]
    frequencies = [[10, 20, 30], [40, 50, 60], [70, 80, 90]]
    model_nm = "lda"

    fig = plot_cloud(corrs, features, model_nm, frequencies=frequencies)
    assert isinstance(fig, mpl.figure.Figure)


def test_plot_surf_maps(tmp_path_factory):
    """Test plotting surface maps."""
    tmpdir = tmp_path_factory.mktemp("test_fetch_spinsamples")

    lh_grad = np.random.rand(32492)
    rh_grad = np.random.rand(32492)

    fig = plot_surf_maps(lh_grad, rh_grad)
    assert isinstance(fig, mpl.figure.Figure)

    # Test with out_fig
    out_fig = tmpdir / "test_surfplot.png"

    plot_surf_maps(lh_grad, rh_grad, out_fig=out_fig)
    assert out_fig.exists()
