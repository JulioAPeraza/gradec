"""Generate fixtures for tests."""

import os

import pytest
from nimare.dataset import Dataset
from nimare.tests.utils import get_test_data_path


@pytest.fixture(scope="session")
def testdata_laird():
    """Load data from dataset into global variables."""
    return Dataset(os.path.join(get_test_data_path(), "neurosynth_laird_studies.json"))


@pytest.fixture(scope="session")
def reduced_testdata_laird():
    """Load data from dataset into global variables."""
    dset = Dataset(os.path.join(get_test_data_path(), "neurosynth_laird_studies.json"))
    dset.annotations = dset.annotations.iloc[:, :100]
    return dset
