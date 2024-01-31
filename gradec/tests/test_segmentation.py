"""Test segmentation module."""

import numpy as np
import pytest

from gradec.segmentation import KMeansSegmentation, PCTSegmentation


@pytest.mark.parametrize(
    "segmentation",
    [
        (PCTSegmentation(10)),
        (KMeansSegmentation(10)),
    ],
)
def test_segmentation_smoke(segmentation):
    """Smoke test for segmentation."""
    gradient = np.random.rand(100)
    segmentation = segmentation.fit(gradient)
    grad_maps = segmentation.transform()

    assert len(grad_maps) == 10
    assert all(len(grad_map) == 100 for grad_map in grad_maps)
