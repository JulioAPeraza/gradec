"""Decode module."""
import os
import os.path as op
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
from nimare.annotate.gclda import GCLDAModel
from nimare.extract import download_abstracts, fetch_neuroquery, fetch_neurosynth
from nimare.io import convert_neurosynth_to_dataset
from nimare.meta.cbma.mkda import MKDAChi2
from nimare.stats import pearson

from gradec.model import _get_counts, annotate_lda
from gradec.utils import _check_ncores


def _get_dataset(dset_nm, basepath):
    data_dir = op.join(basepath, "data")
    os.makedirs(data_dir, exist_ok=True)

    if dset_nm == "neurosynth":
        files = fetch_neurosynth(
            data_dir=data_dir,
            version="7",
            overwrite=False,
            source="abstract",
            vocab="terms",
        )
    elif dset_nm == "neuroquery":
        files = fetch_neuroquery(
            data_dir=data_dir,
            version="1",
            overwrite=False,
            source="combined",
            vocab="neuroquery6308",
            type="tfidf",
        )

    dataset_db = files[0]

    dset = convert_neurosynth_to_dataset(
        coordinates_file=dataset_db["coordinates"],
        metadata_file=dataset_db["metadata"],
        annotations_files=dataset_db["features"],
    )
    dset.update_path(basepath)
    dset = download_abstracts(dset, "jpera054@fiu.edu")


def _train_decoder(dset, dset_nm, model_nm, n_topics, n_cores):
    feature_group = (
        "terms_abstract_tfidf" if dset_nm == "neurosynth" else "neuroquery6308_combined_tfidf"
    )
    frequency_threshold = 0.001 if model_nm == "term" else 0.05

    if (model_nm == "term") or (model_nm == "lda"):
        decoder = CorrelationDecoder(
            frequency_threshold=frequency_threshold,
            meta_estimator=MKDAChi2,
            feature_group=feature_group,
            target_image="z_desc-specificity",
            n_cores=n_cores,
        )
        decoder.fit(dset)
        metamap_arr = decoder.images_
    elif model_nm == "gclda":
        counts_df = _get_counts(dset, dset_nm)

        gclda_model = GCLDAModel(
            counts_df,
            dset.coordinates,
            mask=dset.masker.mask_img,
            n_topics=n_topics,
            n_regions=4,
            symmetric=True,
            n_cores=n_cores,
        )
        gclda_model.fit(n_iters=1000, loglikely_freq=100)

        metamap_arr = gclda_model.p_voxel_g_topic_.T

    return metamap_arr


class Decoder(ABCMeta):
    """Base class for decoders in :mod:`~gradec.decode`."""

    def __init__(
        self,
        model_nm="lda",
        n_topics=200,
        basepath=None,
        n_cores=1,
    ):
        self.model_nm = model_nm
        self.n_topics = n_topics
        self.basepath = op.abspath(basepath) if basepath else op.abspath(".")
        self.n_cores = _check_ncores(n_cores)

    @abstractmethod
    def _fit(self, dset_nm):
        """Apply decoding to dataset and output results.

        Must return a DataFrame, with one row for each feature.
        """

    def fit(self, dset_nm):
        self.dset_nm = dset_nm
        self.dset = _get_dataset(dset_nm, self.basepath)
        if self.model_nm == "lda":
            self.dset = annotate_lda(
                self.dset,
                dset_nm,
                n_topics=self.n_topics,
                n_cores=self.n_cores,
            )

        self.metamap_arr = _train_decoder(
            self.dset,
            self.dset_nm,
            self.model_nm,
            self.n_topics,
            self.n_cores,
        )


class CorrelationDecoder(Decoder):
    """Decode an unthresholded image by correlating the image with meta-analytic maps.

    .. versionchanged:: 0.1.0

        * New method: `load_imgs`. Load pre-generated meta-analytic maps for decoding.

        * New attribute: `results_`. MetaResult object containing masker, meta-analytic maps,
          and tables. This attribute replaces `masker`, `features_`, and `images_`.

    .. versionchanged:: 0.0.13

        * New parameter: `n_cores`. Number of cores to use for parallelization.

    .. versionchanged:: 0.0.12

        * Remove low-memory option in favor of sparse arrays.

    Parameters
    ----------
    feature_group : :obj:`str`, optional
        Feature group
    features : :obj:`list`, optional
        Features
    frequency_threshold : :obj:`float`, optional
        Frequency threshold
    meta_estimator : :class:`~nimare.base.CBMAEstimator`, optional
        Meta-analysis estimator. Default is :class:`~nimare.meta.mkda.MKDAChi2`.
    target_image : :obj:`str`, optional
        Name of meta-analysis results image to use for decoding.
    n_cores : :obj:`int`, optional
        Number of cores to use for parallelization.
        If <=0, defaults to using all available cores.
        Default is 1.

    Warnings
    --------
    Coefficients from correlating two maps have very large degrees of freedom,
    so almost all results will be statistically significant. Do not attempt to
    evaluate results based on significance.
    """

    _required_inputs = {
        "coordinates": ("coordinates", None),
        "annotations": ("annotations", None),
    }

    def __init__(
        self,
        feature_group=None,
        features=None,
        frequency_threshold=0.001,
        meta_estimator=None,
        target_image="z_desc-specificity",
        n_cores=1,
    ):
        self.feature_group = feature_group
        self.features = features
        self.frequency_threshold = frequency_threshold
        self.meta_estimator = meta_estimator
        self.target_image = target_image
        self.n_cores = _check_ncores(n_cores)

    def transform(self, img):
        """Correlate target image with each feature-specific meta-analytic map.

        Parameters
        ----------
        img : :obj:`~nibabel.nifti1.Nifti1Image`
            Image to decode. Must be in same space as ``dataset``.

        Returns
        -------
        out_df : :obj:`pandas.DataFrame`
            DataFrame with one row for each feature, an index named "feature", and one column: "r".
        """
        if not hasattr(self, "results_"):
            raise AttributeError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                "Call 'fit' or 'load_imgs' before using 'transform'."
            )

        # Make sure we return a copy of the MetaResult
        results = self.results_.copy()
        features = list(results.maps.keys())
        images = np.array(list(results.maps.values()))

        img_vec = results.masker.transform(img)
        corrs = pearson(img_vec, images)
        out_df = pd.DataFrame(index=features, columns=["r"], data=corrs)
        out_df.index.name = "feature"

        # Update self.results_ to include the new table
        results.tables["correlation"] = out_df
        self.results_ = results

        return out_df
