"""Fetch data."""
import gzip
import hashlib
import os
import pickle
import shutil
from glob import glob

import numpy as np
import pandas as pd
from nilearn.datasets.utils import _fetch_file, _fetch_files
from nimare.dataset import Dataset
from scipy.sparse import load_npz

from gradec.utils import get_data_dir

OSF_URL = "https://osf.io/{}/download"
OSF_DICT = {
    "term_neurosynth_metamaps.npz": "rt843",
    "term_neurosynth_features.csv": "hyjrk",
    "term_neuroquery_metamaps.npz": "azyw2",
    "term_neuroquery_features.csv": "xtjna",
    "lda_neurosynth_metamaps.npz": "wmp4b",
    "lda_neurosynth_features.csv": "ve3nj",
    "lda_neuroquery_metamaps.npz": "t5fjb",
    "lda_neuroquery_features.csv": "u68w7",
    "gclda_neurosynth_metamaps.npz": "hyknz",
    "gclda_neurosynth_features.csv": "jcrkd",
    "gclda_neurosynth_model.pkl.gz": "bg8ef",
    "gclda_neuroquery_metamaps.npz": "t5k2b",
    "gclda_neuroquery_features.csv": "trcxs",
    "gclda_neuroquery_model.pkl.gz": "vsm65",
    "spinsamples_fslr.npz": "q5yv6",
    "neuroquery_counts": "p39mg",
}


def _get_osf_url(filename):
    osf_id = OSF_DICT[filename]
    return OSF_URL.format(osf_id)


def _mk_tmpdir(data_dir, file, url):
    """Make a temporary directory for fetching."""
    files_pickle = pickle.dumps([(file, url)])
    files_md5 = hashlib.md5(files_pickle).hexdigest()
    temp_dir = os.path.join(data_dir, files_md5)

    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

    return temp_dir


def _my_fetch_file(data_dir, filename, url, overwrite=False, resume=True, verbose=1):
    """Fetch a file from OSF."""
    path_name = os.path.join(data_dir, filename)
    if not os.path.exists(path_name) or overwrite:
        # Fetch file
        tmpdir = _mk_tmpdir(data_dir, filename, url)
        temp_fn = _fetch_file(url, tmpdir, resume=resume, verbose=verbose)

        # Move and delete tmpdir
        shutil.move(temp_fn, path_name)
        shutil.rmtree(tmpdir)

    return path_name


def _fetch_neuroquery_counts(data_dir=None, overwrite=False, resume=True, verbose=1):
    data_dir = get_data_dir(os.path.join(data_dir, "meta-analysis", "neuroquery"))

    filename = "neuroquery_counts"
    url = _get_osf_url(filename)
    opts = dict(uncompress=True, overwrite=overwrite)
    counts_fns = _fetch_files(data_dir, [(filename, url, opts)], resume=resume, verbose=verbose)

    counts_arr_fns = glob(os.path.join(counts_fns[0], "*_features.npz"))
    counts_sparse = sum(load_npz(counts_arr_fn) for counts_arr_fn in counts_arr_fns)

    return np.asarray(counts_sparse.todense())


def _fetch_features(dset_nm, model_nm, data_dir=None, overwrite=False, resume=True, verbose=1):
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
    data_dir = get_data_dir(os.path.join(data_dir, "decoding"))

    filename = f"{model_nm}_{dset_nm}_features.csv"
    url = _get_osf_url(filename)

    features_fn = _my_fetch_file(
        data_dir,
        filename,
        url,
        overwrite=overwrite,
        resume=resume,
        verbose=verbose,
    )

    df = pd.read_csv(features_fn)
    return df.values.tolist()


def _fetch_metamaps(dset_nm, model_nm, data_dir=None, overwrite=False, resume=True, verbose=1):
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
    data_dir = get_data_dir(os.path.join(data_dir, "decoding"))

    filename = f"{model_nm}_{dset_nm}_metamaps.npz"
    url = _get_osf_url(filename)

    metamaps_fn = _my_fetch_file(
        data_dir,
        filename,
        url,
        overwrite=overwrite,
        resume=resume,
        verbose=verbose,
    )

    return np.load(metamaps_fn)["arr"]


def _fetch_spinsamples(data_dir=None, overwrite=False, resume=True, verbose=1):
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

    filename = "spinsamples_fslr.npz"
    url = _get_osf_url(filename)

    spinsamples_fn = _my_fetch_file(
        data_dir,
        filename,
        url,
        overwrite=overwrite,
        resume=resume,
        verbose=verbose,
    )

    return np.load(spinsamples_fn)["arr"]


def _fetch_dataset(dset_nm, data_dir=None, overwrite=False, resume=True, verbose=1):
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

    dset_fn = _my_fetch_file(
        data_dir,
        filename,
        url,
        overwrite=overwrite,
        resume=resume,
        verbose=verbose,
    )

    return Dataset.load(dset_fn)


def _fetch_model(dset_nm, model_nm, data_dir=None, overwrite=False, resume=True, verbose=1):
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

    model_fn = _my_fetch_file(
        data_dir,
        filename,
        url,
        overwrite=overwrite,
        resume=resume,
        verbose=verbose,
    )

    model_file = gzip.open(model_fn, "rb")
    return pickle.load(model_file)


def _fetch_decoder(dset_nm, model_nm, data_dir=None, overwrite=False, resume=True, verbose=1):
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

    decoder_fn = _my_fetch_file(
        data_dir,
        filename,
        url,
        overwrite=overwrite,
        resume=resume,
        verbose=verbose,
    )

    decoder_file = gzip.open(decoder_fn, "rb")
    return pickle.load(decoder_file)
