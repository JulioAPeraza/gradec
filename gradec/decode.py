"""Decode module."""
import os.path as op
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nilearn import masking
from nimare.annotate.gclda import GCLDAModel
from nimare.decode.utils import weight_priors
from nimare.extract import download_abstracts, fetch_neuroquery, fetch_neurosynth
from nimare.io import convert_neurosynth_to_dataset
from nimare.meta.cbma.mkda import MKDAChi2

from gradec.fetcher import (
    _fetch_features,
    _fetch_metamaps,
    _fetch_model,
    _fetch_spinsamples,
)
from gradec.model import _get_counts, annotate_lda
from gradec.stats import _gen_spinsamples, _permtest_pearson
from gradec.transform import _mni152_to_fslr
from gradec.utils import _check_ncores, get_data_dir


def _get_dataset(dset_nm, data_dir):
    data_dir = get_data_dir(op.join(data_dir, "data", dset_nm))

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
    dset.update_path(data_dir)

    if dset_nm == "neurosynth":
        # Download abstracts only for Neurosynth
        dset = download_abstracts(dset, "jpera054@fiu.edu")


class Decoder(metaclass=ABCMeta):
    """Base class for decoders in :mod:`~gradec.decode`."""

    def __init__(
        self,
        model_nm="lda",
        n_topics=200,
        n_top_words=3,
        use_fetchers=True,
        data_dir=None,
        n_cores=1,
    ):
        self.model_nm = model_nm
        self.n_topics = n_topics
        self.n_top_words = n_top_words
        self.use_fetchers = use_fetchers
        self.data_dir = op.abspath(data_dir) if data_dir else op.abspath(".")
        self.neuromaps_dir = get_data_dir(op.join(self.data_dir, "neuromaps"))
        self.n_cores = _check_ncores(n_cores)

    @abstractmethod
    def transform(self, dset_nm):
        """Apply decoding to dataset and output results.

        Must return a DataFrame, with one row for each feature.
        """

    def _get_features(self):
        if self.model_nm in ["term", "lda"]:
            return [f.split("__")[-1] for f in self.decoder.features_]
        vocabulary = np.array(self.model.vocabulary)
        return list(vocabulary)

    def _train_decoder(self):
        feature_group = (
            "terms_abstract_tfidf"
            if self.dset_nm == "neurosynth"
            else "neuroquery6308_combined_tfidf"
        )
        frequency_threshold = 0.001 if self.model_nm == "term" else 0.05

        if self.model_nm == "lda":
            self.dset, self.model = annotate_lda(
                self.dset,
                self.dset_nm,
                n_topics=self.n_topics,
                n_cores=self.n_cores,
            )

        if self.model_nm in ["term", "lda"]:
            decoder = CorrelationDecoder(
                frequency_threshold=frequency_threshold,
                meta_estimator=MKDAChi2,
                feature_group=feature_group,
                target_image="z_desc-specificity",
                n_cores=self.n_cores,
            )
            decoder.fit(self.dset)
            self.decoder = decoder

            metamaps_arr = decoder.images_
            metamaps = decoder.masker.inverse_transform(metamaps_arr)

        elif self.model_nm == "gclda":
            counts_df = _get_counts(self.dset, self.dset_nm)

            gclda_model = GCLDAModel(
                counts_df,
                self.dset.coordinates,
                mask=self.dset.masker.mask_img,
                n_topics=self.n_topics,
                n_regions=4,
                symmetric=True,
                n_cores=self.n_cores,
            )
            gclda_model.fit(n_iters=1000, loglikely_freq=100)
            self.model = gclda_model

            metamaps_arr = gclda_model.p_topic_g_voxel_.T
            metamaps = masking.unmask(metamaps_arr, gclda_model.mask)

        return metamaps

    def fit(self, dset_nm):
        """Fit decoder to dataset."""
        self.dset_nm = dset_nm
        if self.use_fetchers:
            metamaps_fslr = _fetch_metamaps(self.dset_nm, self.model_nm, self.data_dir)
            features = _fetch_features(self.dset_nm, self.model_nm, self.data_dir)
            spinsamples_fslr = _fetch_spinsamples(self.data_dir)
        else:
            self.dset = _get_dataset(dset_nm, self.data_dir)
            metamaps = self._train_decoder()

            metamaps_fslr = _mni152_to_fslr(metamaps, self.neuromaps_dir)
            features = self._get_features()
            spinsamples_fslr = _gen_spinsamples(self.neuromaps_dir, self.n_samples, self.n_cores)

        self.maps_ = metamaps_fslr
        if self.model_nm in ["lda", "gclda"]:
            self.features_ = ["_".join(feature[: self.n_top_words]) for feature in features]
        else:
            self.features_ = [feature[0] for feature in features]

        self.spinsamples_ = spinsamples_fslr


class CorrelationDecoder(Decoder):
    """Decode an unthresholded image by correlating the image with meta-analytic maps."""

    def transform(self, grad_maps):
        """Correlate target image with each feature-specific meta-analytic map.

        Parameters
        ----------
        grad_maps : :obj:`~nibabel.nifti1.Nifti1Image`
            Image to decode. Must be in same space as ``dataset``.

        Returns
        -------
        out_df : :obj:`pandas.DataFrame`
            DataFrame with one row for each feature, an index named "feature", and one column: "r".
        """
        corrs_lst, pvals_lst, corr_pvals_lst = zip(
            *Parallel(n_jobs=self.n_cores)(
                delayed(_permtest_pearson)(grad_map, self.maps_, self.spinsamples_)
                for grad_map in grad_maps
            )
        )

        corrs_data = np.array(corrs_lst).T
        pvals_data = np.array(pvals_lst).T
        corr_pvals_data = np.array(corr_pvals_lst).T

        corrs_df = pd.DataFrame(index=self.features_, data=corrs_data)
        pvals_df = pd.DataFrame(index=self.features_, data=pvals_data)
        corr_pvals_df = pd.DataFrame(index=self.features_, data=corr_pvals_data)
        corrs_df.index.name = "feature"
        pvals_df.index.name = "feature"
        corr_pvals_df.index.name = "feature"

        return corrs_df, pvals_df, corr_pvals_df


class GCLDADecoder(Decoder):
    """Decode an unthresholded image by correlating the image with meta-analytic maps."""

    def transform(self, grad_maps, topic_priors=None, prior_weight=1):
        """Correlate target image with each feature-specific meta-analytic map.

        Parameters
        ----------
        grad_maps : :obj:`~nibabel.nifti1.Nifti1Image`
            Image to decode. Must be in same space as ``dataset``.

        Returns
        -------
        out_df : :obj:`pandas.DataFrame`
            DataFrame with one row for each feature, an index named "feature", and one column: "r".
        """
        model = _fetch_model(self.dset_nm, self.model_nm, data_dir=self.data_dir)

        word_weights = []
        for grad_map in grad_maps:
            topic_weights = np.squeeze(np.dot(self.maps_, grad_map))
            if topic_priors is not None:
                weighted_priors = weight_priors(topic_priors, prior_weight)
                topic_weights *= weighted_priors

            word_weights.append(np.dot(model.p_word_g_topic_, topic_weights))

        word_weights = np.array(word_weights).T

        decoded_df = pd.DataFrame(index=model.vocabulary, data=word_weights)
        decoded_df.index.name = "Term"
        return decoded_df
