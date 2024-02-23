"""Decode module."""

import os.path as op
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nilearn import masking
from nimare import decode
from nimare.annotate.gclda import GCLDAModel
from nimare.decode.utils import weight_priors
from nimare.extract import download_abstracts, fetch_neuroquery, fetch_neurosynth
from nimare.io import convert_neurosynth_to_dataset
from nimare.meta.cbma.mkda import MKDAChi2
from tqdm.auto import tqdm

from gradec.fetcher import (
    _fetch_features,
    _fetch_metamaps,
    _fetch_model,
    _fetch_spinsamples,
)
from gradec.model import _get_counts, annotate_lda
from gradec.stats import _permtest_pearson
from gradec.transform import _vol_to_surf
from gradec.utils import _check_ncores, _conform_features, get_data_dir


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


class Decoder(metaclass=ABCMeta):
    """Base class for decoders in :mod:`~gradec.decode`."""

    def __init__(
        self,
        space="fsLR",
        density="32k",
        feature_group=None,
        use_fetchers=True,
        calc_pvals=True,
        data_dir=None,
        n_cores=1,
    ):
        self.space = space
        self.density = density
        self.feature_group = feature_group
        self.use_fetchers = use_fetchers
        self.calc_pvals = calc_pvals
        self.data_dir = op.abspath(data_dir) if data_dir else op.abspath(".")
        self.neuromaps_dir = get_data_dir(op.join(self.data_dir, "neuromaps"))
        self.n_cores = _check_ncores(n_cores)

        self.n_top_words = None  # Set in child classes

    @abstractmethod
    def transform(self, dset_nm):
        """Apply decoding to dataset and output a DF."""

    @abstractmethod
    def _train_decoder(self, dset_nm):
        """Train decoder on dataset."""

    @abstractmethod
    def _get_features(self, dset_nm):
        """Get features from dataset."""

    def fit(self, dset_nm, dset=None):
        """Fit decoder to dataset."""
        self.dset_nm = dset_nm

        if self.use_fetchers:
            metamaps_surf = _fetch_metamaps(
                self.dset_nm,
                self.model_nm,
                space=self.space,
                density=self.density,
                data_dir=self.data_dir,
            )
            features = _fetch_features(self.dset_nm, self.model_nm, data_dir=self.data_dir)

        else:
            self.dset = dset or _get_dataset(dset_nm, self.data_dir)
            metamaps = self._train_decoder()

            metamaps_surf = _vol_to_surf(
                metamaps,
                space=self.space,
                density=self.density,
                neuromaps_dir=self.neuromaps_dir,
            )
            features = self._get_features()

            # This feature is desabled for now because it takes too long.
            """
            if self.calc_pvals:
                spinsamples_surf = _gen_spinsamples(
                    self.space,
                    self.density,
                    self.neuromaps_dir,
                    self.n_samples,
                    self.n_cores,
                )
            """

        if self.calc_pvals:
            spinsamples_surf = _fetch_spinsamples(
                n_samples=self.n_samples,
                space=self.space,
                density=self.density,
                data_dir=self.data_dir,
            )
            self.spinsamples_ = spinsamples_surf
        else:
            self.spinsamples_ = None

        self.maps_ = metamaps_surf
        self.features_ = _conform_features(features, self.model_nm, self.n_top_words)


class CorrelationDecoder(Decoder):
    """Decode an unthresholded image by correlating the image with meta-analytic maps."""

    def __init__(self, n_samples=1000, **kwargs):
        super().__init__(**kwargs)
        self.n_samples = n_samples

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
        results_lst = tqdm(
            zip(
                *Parallel(return_as="generator", n_jobs=self.n_cores)(
                    delayed(_permtest_pearson)(grad_map, self.maps_, self.spinsamples_)
                    for grad_map in grad_maps
                ),
            ),
            total=len(grad_maps),
        )

        data_df_lst = []
        for results in results_lst:
            data = np.array(results).T
            data_df = pd.DataFrame(index=self.features_, data=data)
            data_df.index.name = "feature"

            if self.calc_pvals:
                data_df_lst.append(data_df)
            else:
                return data_df

        return data_df_lst


class TermDecoder(CorrelationDecoder):
    """Decode an unthresholded image by correlating the image with meta-analytic maps."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_nm = "term"

    def _get_features(self):
        decoder_features = list(self.decoder.results_.maps.keys())
        return [f.split("__")[-1] for f in decoder_features]

    def _train_decoder(self):
        frequency_threshold = 0.001
        decoder = decode.CorrelationDecoder(
            frequency_threshold=frequency_threshold,
            meta_estimator=MKDAChi2,
            feature_group=self.feature_group,
            target_image="z_desc-association",
            n_cores=self.n_cores,
        )
        decoder.fit(self.dset)
        self.decoder = decoder

        metamaps_arr = np.array(list(decoder.results_.maps.values()))
        return decoder.results_.masker.inverse_transform(metamaps_arr)

    def transform(self, grad_maps, method="correlation", topic_priors=None, prior_weight=1):
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
        if method == "correlation":
            return super().transform(grad_maps)


class LDADecoder(CorrelationDecoder):
    """Decode an unthresholded image by correlating the image with meta-analytic maps."""

    def __init__(self, n_topics=200, n_top_words=3, **kwargs):
        super().__init__(**kwargs)
        self.n_topics = n_topics
        self.n_top_words = n_top_words
        self.model_nm = "lda"

    def _get_features(self):
        topic_word_weights = self.model.distributions_["p_topic_g_word"]
        sorted_weights_idxs = np.argsort(-topic_word_weights, axis=1)
        vocabulary = self.model.distributions_["p_topic_g_word_df"].columns.to_numpy()

        n_topics = topic_word_weights.shape[0]
        features = [vocabulary[sorted_weights_idxs[topic_i, :]] for topic_i in range(n_topics)]

        # Get only the topics that were used to trained the decoder (those that
        # have at least one study over the 0.05 threshold)
        decoder_features = list(self.decoder.results_.maps.keys())
        feat_used_ids = [int(f.split("__")[-1].split("_")[0]) - 1 for f in decoder_features]

        return [features[i] for i in feat_used_ids]

    def _train_decoder(self):
        if ("abstract" not in self.dset.texts) and (self.dset_nm == "neurosynth"):
            # Download abstracts only for Neurosynth
            self.dset = download_abstracts(self.dset, "jpera054@fiu.edu")

        self.dset, self.model = annotate_lda(
            self.dset,
            self.dset_nm,
            self.feature_group,
            n_topics=self.n_topics,
            n_cores=self.n_cores,
        )

        frequency_threshold = 0.05
        new_feature_group = f"LDA{self.n_topics}"
        decoder = decode.CorrelationDecoder(
            frequency_threshold=frequency_threshold,
            meta_estimator=MKDAChi2,
            feature_group=new_feature_group,
            target_image="z_desc-association",
            n_cores=self.n_cores,
        )
        decoder.fit(self.dset)
        self.decoder = decoder

        metamaps_arr = np.array(list(decoder.results_.maps.values()))
        return decoder.results_.masker.inverse_transform(metamaps_arr)

    def transform(self, grad_maps, method="correlation", topic_priors=None, prior_weight=1):
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
        if method == "correlation":
            return super().transform(grad_maps)


class GCLDADecoder(CorrelationDecoder):
    """Decode an unthresholded image by correlating the image with meta-analytic maps."""

    def __init__(self, n_iters=1000, n_topics=200, n_top_words=3, **kwargs):
        super().__init__(**kwargs)
        self.n_iters = n_iters
        self.n_topics = n_topics
        self.n_top_words = n_top_words
        self.model_nm = "gclda"

    def _get_features(self):
        topic_word_weights = self.model.p_word_g_topic_.T
        sorted_weights_idxs = np.argsort(-topic_word_weights, axis=1)
        vocabulary = np.array(self.model.vocabulary)

        n_topics = topic_word_weights.shape[0]
        return [vocabulary[sorted_weights_idxs[topic_i, :]] for topic_i in range(n_topics)]

    def _train_decoder(self):
        counts_df = _get_counts(self.dset, self.dset_nm, self.feature_group)
        gclda_model = GCLDAModel(
            counts_df,
            self.dset.coordinates,
            mask=self.dset.masker.mask_img,
            n_topics=self.n_topics,
            n_regions=4,
            symmetric=True,
        )
        gclda_model.fit(n_iters=self.n_iters, loglikely_freq=10)
        self.model = gclda_model

        metamaps_arr = gclda_model.p_topic_g_voxel_.T
        return masking.unmask(metamaps_arr, gclda_model.mask)

    def transform(self, grad_maps, method="gclda", topic_priors=None, prior_weight=1):
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
        if method == "correlation":
            return super().transform(grad_maps)

        model = (
            _fetch_model(self.dset_nm, self.model_nm, data_dir=self.data_dir)
            if self.use_fetchers
            else self.model
        )

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
