"""Test segmentation module."""

import numpy as np
import pytest

from gradec.segmentation import KMeansSegmentation, PCTSegmentation


@pytest.mark.parametrize(
    "segmentation",
    [
        (PCTSegmentation(n_segments=10)),
        (KMeansSegmentation(n_segments=10)),
    ],
    ids=["PCT", "KMeans"],
)
def test_segmentation_smoke(segmentation):
    """Smoke test for segmentation."""
    gradient = np.random.rand(100)
    segmentation = segmentation.fit(gradient)
    grad_maps = segmentation.transform()

    assert len(grad_maps) == 10
    assert all(len(grad_map) == 100 for grad_map in grad_maps)


def test_segmentation_kmeans_highdim():
    """Test segmentation with high-dimensional gradient."""
    gradients = np.random.rand(100, 4)

    segmentation = KMeansSegmentation(n_segments=10).fit(gradients)
    grad_maps = segmentation.transform()

    assert len(grad_maps) == 10
    assert all(len(grad_map) == 100 for grad_map in grad_maps)


def test_segmentation_pct_highdim():
    """Test segmentation with high-dimensional gradient."""
    gradients = np.random.rand(100, 10)

    with pytest.raises(ValueError):
        PCTSegmentation(n_segments=10).fit(gradients)
