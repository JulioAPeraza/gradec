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
    "source-neuroquery_desc-gclda_features.csv": "trcxs",
    "source-neuroquery_desc-gclda_frequencies.csv": "kcem9",
    "source-neuroquery_desc-gclda_classification.csv": "93dvg",
    "source-neuroquery_desc-gclda_space-civet_density-41k_metamaps.npz": "qbn7g",
    "source-neuroquery_desc-gclda_space-fsLR_density-32k_metamaps.npz": "ey6cw",
    "source-neuroquery_desc-gclda_space-fsLR_density-164k_metamaps.npz": "4erpq",
    "source-neuroquery_desc-gclda_space-fsaverage_density-3k_metamaps.npz": "jg34m",
    "source-neuroquery_desc-gclda_space-fsaverage_density-10k_metamaps.npz": "sh9eq",
    "source-neuroquery_desc-gclda_space-fsaverage_density-41k_metamaps.npz": "ca2hp",
    "source-neuroquery_desc-gclda_space-fsaverage_density-164k_metamaps.npz": "4ne7f",
    "source-neuroquery_desc-lda_features.csv": "u68w7",
    "source-neuroquery_desc-lda_frequencies.csv": "ygpxf",
    "source-neuroquery_desc-lda_classification.csv": "mtwvc",
    "source-neuroquery_desc-lda_space-civet_density-41k_metamaps.npz": "74trb",
    "source-neuroquery_desc-lda_space-fsLR_density-32k_metamaps.npz": "k4xza",
    "source-neuroquery_desc-lda_space-fsLR_density-164k_metamaps.npz": "wvznp",
    "source-neuroquery_desc-lda_space-fsaverage_density-3k_metamaps.npz": "xbrgd",
    "source-neuroquery_desc-lda_space-fsaverage_density-41k_metamaps.npz": "hu3y4",
    "source-neuroquery_desc-lda_space-fsaverage_density-10k_metamaps.npz": "qwb5u",
    "source-neuroquery_desc-lda_space-fsaverage_density-164k_metamaps.npz": "qzhrn",
    "source-neuroquery_desc-term_features.csv": "xtjna",
    "source-neuroquery_desc-term_classification.csv": "ypqzx",
    "source-neuroquery_desc-term_space-civet_density-41k_metamaps.npz": "vw9fr",
    "source-neuroquery_desc-term_space-fsLR_density-32k_metamaps.npz": "38fh4",
    "source-neuroquery_desc-term_space-fsLR_density-164k_metamaps.npz": "",  # Updt
    "source-neuroquery_desc-term_space-fsaverage_density-3k_metamaps.npz": "b5jme",
    "source-neuroquery_desc-term_space-fsaverage_density-10k_metamaps.npz": "dt3j4",
    "source-neuroquery_desc-term_space-fsaverage_density-41k_metamaps.npz": "9xsqb",
    "source-neuroquery_desc-term_space-fsaverage_density-164k_metamaps.npz": "",  # Updt
    "source-neurosynth_desc-gclda_features.csv": "jcrkd",
    "source-neurosynth_desc-gclda_frequencies.csv": "48hbz",
    "source-neurosynth_desc-gclda_classification.csv": "p7nd9",
    "source-neurosynth_desc-gclda_space-civet_density-41k_metamaps.npz": "swp3c",
    "source-neurosynth_desc-gclda_space-fsLR_density-32k_metamaps.npz": "hwdft",
    "source-neurosynth_desc-gclda_space-fsLR_density-164k_metamaps.npz": "qzyvj",
    "source-neurosynth_desc-gclda_space-fsaverage_density-3k_metamaps.npz": "r5p7x",
    "source-neurosynth_desc-gclda_space-fsaverage_density-10k_metamaps.npz": "w2xj3",
    "source-neurosynth_desc-gclda_space-fsaverage_density-41k_metamaps.npz": "dw3t9",
    "source-neurosynth_desc-gclda_space-fsaverage_density-164k_metamaps.npz": "mw2pa",
    "source-neurosynth_desc-lda_features.csv": "ve3nj",
    "source-neurosynth_desc-lda_frequencies.csv": "485je",
    "source-neurosynth_desc-lda_classification.csv": "9mrxb",
    "source-neurosynth_desc-lda_space-civet_density-41k_metamaps.npz": "86u59",
    "source-neurosynth_desc-lda_space-fsLR_density-32k_metamaps.npz": "9ftkm",
    "source-neurosynth_desc-lda_space-fsLR_density-164k_metamaps.npz": "bxefz",
    "source-neurosynth_desc-lda_space-fsaverage_density-3k_metamaps.npz": "pb9mw",
    "source-neurosynth_desc-lda_space-fsaverage_density-10k_metamaps.npz": "urg3a",
    "source-neurosynth_desc-lda_space-fsaverage_density-41k_metamaps.npz": "dbv4z",
    "source-neurosynth_desc-lda_space-fsaverage_density-164k_metamaps.npz": "7rw8c",
    "source-neurosynth_desc-term_features.csv": "hyjrk",
    "source-neurosynth_desc-term_classification.csv": "sd4wy",
    "source-neurosynth_desc-term_space-civet_density-41k_metamaps.npz": "rwxta",
    "source-neurosynth_desc-term_space-fsLR_density-32k_metamaps.npz": "ju2tk",
    "source-neurosynth_desc-term_space-fsLR_density-164k_metamaps.npz": "dzm39",
    "source-neurosynth_desc-term_space-fsaverage_density-3k_metamaps.npz": "hvw6x",
    "source-neurosynth_desc-term_space-fsaverage_density-10k_metamaps.npz": "q2txn",
    "source-neurosynth_desc-term_space-fsaverage_density-41k_metamaps.npz": "cyuqn",
    "source-neurosynth_desc-term_space-fsaverage_density-164k_metamaps.npz": "s8txr",
    "source-vasa2018_desc-null1000_space-civet_density-41k_metamaps.npz": "8svr2",  # Updt
    "source-vasa2018_desc-null1000_space-fsLR_density-32k_spinsamples.npz": "q5yv6",
    "source-vasa2018_desc-null1000_space-fsLR_density-164k_spinsamples.npz": "nj3cr",  # Updt
    "source-vasa2018_desc-null1000_space-fsaverage_density-3k_spinsamples.npz": "8vjxm",  # Updt
    "source-vasa2018_desc-null1000_space-fsaverage_density-10k_spinsamples.npz": "yz59g",  # Updt
    "source-vasa2018_desc-null1000_space-fsaverage_density-41k_spinsamples.npz": "esr9y",  # Updt
    "source-vasa2018_desc-null1000_space-fsaverage_density-164k_spinsamples.npz": "e956f",  # Updt
    "gclda_neurosynth_model.pkl.gz": "bg8ef",
    "gclda_neuroquery_model.pkl.gz": "vsm65",
    "lda_neurosynth_model.pkl.gz": "3kgfe",
    "lda_neuroquery_model.pkl.gz": "wevdn",
    "hcp-s1200_gradients.npy": "t95gk",
    "principal_gradient.npy": "5th7c",
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
    data_dir = get_data_dir(data_dir)
    counts_dir = get_data_dir(os.path.join(data_dir, "meta-analysis", "neuroquery"))

    filename = "neuroquery_counts"
    url = _get_osf_url(filename)
    opts = dict(uncompress=True, overwrite=overwrite)
    counts_fns = _fetch_files(counts_dir, [(filename, url, opts)], resume=resume, verbose=verbose)

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
    data_dir = get_data_dir(data_dir)
    dec_dir = get_data_dir(os.path.join(data_dir, "decoding"))

    filename = f"source-{dset_nm}_desc-{model_nm}_features.csv"
    url = _get_osf_url(filename)

    features_fn = _my_fetch_file(
        dec_dir,
        filename,
        url,
        overwrite=overwrite,
        resume=resume,
        verbose=verbose,
    )

    df = pd.read_csv(features_fn)
    return df.values.tolist()


def _fetch_frequencies(dset_nm, model_nm, data_dir=None, overwrite=False, resume=True, verbose=1):
    """Fetch frequencies from OSF.

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
    :class:`list` of list
        List of frequency values.
    """
    data_dir = get_data_dir(data_dir)
    dec_dir = get_data_dir(os.path.join(data_dir, "decoding"))

    filename = f"source-{dset_nm}_desc-{model_nm}_frequencies.csv"
    url = _get_osf_url(filename)

    features_fn = _my_fetch_file(
        dec_dir,
        filename,
        url,
        overwrite=overwrite,
        resume=resume,
        verbose=verbose,
    )

    df = pd.read_csv(features_fn)
    return df.values.tolist()


def _fetch_classification(
    dset_nm,
    model_nm,
    data_dir=None,
    overwrite=False,
    resume=True,
    verbose=1,
):
    """Fetch classification from OSF.

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
    :class:`list` of list
        List of classification values.
    """
    data_dir = get_data_dir(data_dir)
    dec_dir = get_data_dir(os.path.join(data_dir, "decoding"))

    filename = f"source-{dset_nm}_desc-{model_nm}_classification.csv"
    url = _get_osf_url(filename)

    features_fn = _my_fetch_file(
        dec_dir,
        filename,
        url,
        overwrite=overwrite,
        resume=resume,
        verbose=verbose,
    )

    df = pd.read_csv(features_fn, index_col="Classification")
    return df.index.tolist(), df.values.tolist()


def _fetch_metamaps(
    dset_nm,
    model_nm,
    space="fsLR",
    density="32k",
    data_dir=None,
    overwrite=False,
    resume=True,
    verbose=1,
):
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
    dec_dir = get_data_dir(os.path.join(data_dir, "decoding"))

    filename = f"source-{dset_nm}_desc-{model_nm}_space-{space}_density-{density}_metamaps.npz"
    url = _get_osf_url(filename)

    metamaps_fn = _my_fetch_file(
        dec_dir,
        filename,
        url,
        overwrite=overwrite,
        resume=resume,
        verbose=verbose,
    )

    return np.load(metamaps_fn)["arr"]


def _fetch_spinsamples(
    n_samples=1000,
    space="fsLR",
    density="32k",
    data_dir=None,
    overwrite=False,
    resume=True,
    verbose=1,
):
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

    filename = f"source-vasa2018_desc-null1000_space-{space}_density-{density}_spinsamples.npz"
    url = _get_osf_url(filename)

    spinsamples_fn = _my_fetch_file(
        data_dir,
        filename,
        url,
        overwrite=overwrite,
        resume=resume,
        verbose=verbose,
    )

    return np.load(spinsamples_fn)["arr"][:, :n_samples]


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
    data_dir = get_data_dir(data_dir)
    dset_dir = get_data_dir(os.path.join(data_dir, "meta-analysis", dset_nm))

    filename = f"{dset_nm}_dataset.pkl.gz"
    url = _get_osf_url(filename)

    dset_fn = _my_fetch_file(
        dset_dir,
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
    data_dir = get_data_dir(data_dir)
    model_dir = get_data_dir(os.path.join(data_dir, "models"))

    filename = f"{model_nm}_{dset_nm}_model.pkl.gz"
    url = _get_osf_url(filename)

    model_fn = _my_fetch_file(
        model_dir,
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
    data_dir = get_data_dir(data_dir)
    dec_dir = get_data_dir(os.path.join(data_dir, "decoding"))

    filename = f"{model_nm}_{dset_nm}_decoder.pkl.gz"
    url = _get_osf_url(filename)

    decoder_fn = _my_fetch_file(
        dec_dir,
        filename,
        url,
        overwrite=overwrite,
        resume=resume,
        verbose=verbose,
    )

    decoder_file = gzip.open(decoder_fn, "rb")
    return pickle.load(decoder_file)


def _fetch_gradients(cortical=False, data_dir=None, overwrite=False, resume=True, verbose=1):
    """Fetch gradients from OSF.

    Parameters
    ----------
    cortical : :obj:`bool`, optional
        Whether to fetch cortical or whole-brain gradients.
        Default=False.
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
    :obj:`numpy.ndarray`
        2D array of gradients.
    """
    N_VERTICES = 59412  # Number of vertices in fsLR 32k space for HCP S1200
    data_dir = get_data_dir(data_dir)
    grad_dir = get_data_dir(os.path.join(data_dir, "gradient"))

    filename = "hcp-s1200_gradients.npy"
    url = _get_osf_url(filename)

    gradient_fn = _my_fetch_file(
        grad_dir,
        filename,
        url,
        overwrite=overwrite,
        resume=resume,
        verbose=verbose,
    )
    gradients = np.load(gradient_fn)

    return gradients[:N_VERTICES] if cortical else gradients


def _fetch_principal_gradients(data_dir=None, overwrite=False, resume=True, verbose=1):
    """Fetch principal gradient from OSF.

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
    :obj:`numpy.ndarray`
        1D array, principal gradient.
    """
    data_dir = get_data_dir(data_dir)
    grad_dir = get_data_dir(os.path.join(data_dir, "gradient"))

    filename = "principal_gradient.npy"
    url = _get_osf_url(filename)

    gradient_fn = _my_fetch_file(
        grad_dir,
        filename,
        url,
        overwrite=overwrite,
        resume=resume,
        verbose=verbose,
    )

    return np.load(gradient_fn)
