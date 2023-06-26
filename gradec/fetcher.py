"""Fetch data"""
import gzip
import os
import pickle
from glob import glob

import numpy as np
import pandas as pd
from nilearn.datasets.utils import _fetch_file
from nimare.dataset import Dataset
from scipy.sparse import load_npz

from gradec.utils import get_data_dir

OSF_URL = "https://osf.io/{}/download"
OSF_DICT = {
    "neurosynth_dataset.pkl.gz": "",
    "neuroquery_dataset.pkl.gz": "",
    "term_neurosynth_metamaps.npy": "",
    "term_neurosynth_features.csv": "hyjrk",
    "term_neurosynth_decoder.pkl.gz": "fkldfk2",
    "term_neuroquery_metamaps.npy": "",
    "term_neuroquery_features.csv": "xtjna",
    "term_neuroquery_decoder.pkl.gz": "",
    "lda_neurosynth_metamaps.npy": "",
    "lda_neurosynth_features.csv": "ve3nj",
    "lda_neurosynth_model.pkl.gz": "",
    "lda_neurosynth_decoder.pkl.gz": "",
    "lda_neuroquery_metamaps.npy": "",
    "lda_neuroquery_features.csv": "u68w7",
    "lda_neuroquery_model.pkl.gz": "",
    "lda_neuroquery_decoder.pkl.gz": "",
    "gclda_neurosynth_metamaps.npy": "",
    "gclda_neurosynth_features.csv": "jcrkd",
    "gclda_neurosynth_model.pkl.gz": "",
    "gclda_neuroquery_metamaps.npy": "",
    "gclda_neuroquery_features.csv": "trcxs",
    "gclda_neuroquery_model.pkl.gz": "",
    "spinsamples_fslr.npy": "nukdt",
}


def _get_osf_url(filename):
    osf_id = OSF_DICT[filename]
    return OSF_URL.format(osf_id)


def _fetch_neuroquery_counts(data_dir=None):
    data_dir = get_data_dir(data_dir)
    counts_dir = os.path.join(data_dir, "meta-analysis", "neuroquery", "neuroquery_counts")
    os.makedirs(counts_dir, exist_ok=True)

    counts_arr_fns = glob(os.path.join(counts_dir, "*_features.npz"))
    counts_sparse = None
    for file_i, counts_arr_fn in enumerate(counts_arr_fns):
        if file_i == 0:
            counts_sparse = load_npz(counts_arr_fn)
        else:
            counts_sparse = counts_sparse + load_npz(counts_arr_fn)

    return counts_sparse.todense()


def _fetch_features(dset_nm, model_nm, data_dir=None, resume=True, verbose=1):
    """Fetch features from OSF.

    Parameters
    ----------
    dset_nm : :obj:`str`
        Name of dataset.
    model_nm : :obj:`str`
        Name of model.
    data_dir : :obj:`pathlib.Path` or :obj:`str`, optional
        Path where data should be downloaded. By default,
        files are downloaded in home directory
    resume : :obj:`bool`, optional
        Whether to resume download of a partly-downloaded file.
        Default=True.
    verbose : :obj:`int`, optional
        Verbosity level (0 means no message).
        Default=1.

    Returns
    -------
    :class:`list` of str
        List of feature names.
    """
    data_dir = get_data_dir(data_dir)

    filename = f"{model_nm}_{dset_nm}_features.csv"
    url = _get_osf_url(filename)
    features_fn = os.path.join(data_dir, filename)

    temp_fn = _fetch_file(url, data_dir, resume=resume, verbose=verbose)
    os.rename(temp_fn, features_fn)
    df = pd.read_csv(features_fn)

    return df.values.tolist()


def _fetch_nullmaps(dset_nm, model_nm, data_dir=None, resume=True, verbose=1):
    """Fetch meta-analytic maps from OSF.

    Parameters
    ----------
    dset_nm : :obj:`str`
        Name of dataset.
    model_nm : :obj:`str`
        Name of model.
    data_dir : :obj:`pathlib.Path` or :obj:`str`, optional
        Path where data should be downloaded. By default,
        files are downloaded in home directory
    resume : :obj:`bool`, optional
        Whether to resume download of a partly-downloaded file.
        Default=True.
    verbose : :obj:`int`, optional
        Verbosity level (0 means no message).
        Default=1.

    Returns
    -------
    :class:`list` of str
        List of feature names.
    """
    data_dir = get_data_dir(data_dir)

    filename = f"{model_nm}_{dset_nm}_metamaps.npy"
    url = _get_osf_url(filename)
    metamaps_fn = os.path.join(data_dir, filename)

    temp_fn = _fetch_file(url, data_dir, resume=resume, verbose=verbose)
    os.rename(temp_fn, metamaps_fn)

    return np.load(metamaps_fn)


def _fetch_spinsamples(data_dir=None, resume=True, verbose=1):
    """Fetch spin samples from OSF.

    Parameters
    ----------
    data_dir : :obj:`pathlib.Path` or :obj:`str`, optional
        Path where data should be downloaded. By default,
        files are downloaded in home directory
    resume : :obj:`bool`, optional
        Whether to resume download of a partly-downloaded file.
        Default=True.
    verbose : :obj:`int`, optional
        Verbosity level (0 means no message).
        Default=1.

    Returns
    -------
    (C x V) :class:`numpy.ndarray`
        Array of null samples, where C is the number of samples and V is the number of vertices.
    """
    data_dir = get_data_dir(data_dir)

    filename = "spinsamples.npy"
    url = _get_osf_url(filename)
    nullsamples_fn = os.path.join(data_dir, filename)

    temp_fn = _fetch_file(url, data_dir, resume=resume, verbose=verbose)
    os.rename(temp_fn, nullsamples_fn)

    return np.load(nullsamples_fn)


def _fetch_dataset(dset_nm, data_dir=None, resume=True, verbose=1):
    """Fetch NiMARE dataset object from OSF.

    Parameters
    ----------
    dset_nm : :obj:`str`
        Name of dataset.
    data_dir : :obj:`pathlib.Path` or :obj:`str`, optional
        Path where data should be downloaded. By default,
        files are downloaded in home directory
    resume : :obj:`bool`, optional
        Whether to resume download of a partly-downloaded file.
        Default=True.
    verbose : :obj:`int`, optional
        Verbosity level (0 means no message).
        Default=1.

    Returns
    -------
    :obj:`~nimare.dataset.Dataset`
        NiMARE Dataset object.
    """
    data_dir = get_data_dir(os.path.join(data_dir, "meta-analysis", dset_nm))

    filename = f"{dset_nm}_dataset.pkl.gz"
    url = _get_osf_url(filename)
    dset_fn = os.path.join(data_dir, filename)

    temp_fn = _fetch_file(url, data_dir, resume=resume, verbose=verbose)
    os.rename(temp_fn, dset_fn)

    return Dataset.load(dset_fn)


def _fetch_model(dset_nm, model_nm, data_dir=None, resume=True, verbose=1):
    """Fetch features from OSF.

    Parameters
    ----------
    dset_nm : :obj:`str`
        Name of dataset.
    model_nm : :obj:`str`
        Name of model.
    data_dir : :obj:`pathlib.Path` or :obj:`str`, optional
        Path where data should be downloaded. By default,
        files are downloaded in home directory
    resume : :obj:`bool`, optional
        Whether to resume download of a partly-downloaded file.
        Default=True.
    verbose : :obj:`int`, optional
        Verbosity level (0 means no message).
        Default=1.

    Returns
    -------
    :obj:`~nimare.dataset.Dataset`
        NiMARE LDA or GCLDA object.
    """
    data_dir = get_data_dir(os.path.join(data_dir, "models"))

    filename = f"{model_nm}_{dset_nm}_model.pkl.gz"
    url = _get_osf_url(filename)
    model_fn = os.path.join(data_dir, filename)

    temp_fn = _fetch_file(url, data_dir, resume=resume, verbose=verbose)
    os.rename(temp_fn, model_fn)

    model_file = gzip.open(model_fn, "rb")
    return pickle.load(model_file)


def _fetch_decoder(dset_nm, model_nm, data_dir=None, resume=True, verbose=1):
    """Fetch features from OSF.

    Parameters
    ----------
    dset_nm : :obj:`str`
        Name of dataset.
    model_nm : :obj:`str`
        Name of model.
    data_dir : :obj:`pathlib.Path` or :obj:`str`, optional
        Path where data should be downloaded. By default,
        files are downloaded in home directory
    resume : :obj:`bool`, optional
        Whether to resume download of a partly-downloaded file.
        Default=True.
    verbose : :obj:`int`, optional
        Verbosity level (0 means no message).
        Default=1.

    Returns
    -------
    :obj:`~nimare.dataset.Dataset`
        NiMARE LDA or GCLDA object.
    """
    data_dir = get_data_dir(os.path.join(data_dir, "decoding"))

    filename = f"{model_nm}_{dset_nm}_decoder.pkl.gz"
    url = _get_osf_url(filename)
    decoder_fn = os.path.join(data_dir, filename)

    temp_fn = _fetch_file(url, data_dir, resume=resume, verbose=verbose)
    os.rename(temp_fn, decoder_fn)

    decoder_file = gzip.open(decoder_fn, "rb")
    return pickle.load(decoder_file)


def _fetch_metamaps(dset_nm, model_nm, data_dir=None):
    pass
