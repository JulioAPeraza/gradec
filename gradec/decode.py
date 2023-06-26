"""Decode module."""
import os
import os.path as op
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
from nilearn import masking
from nimare.annotate.gclda import GCLDAModel
from nimare.extract import download_abstracts, fetch_neuroquery, fetch_neurosynth
from nimare.io import convert_neurosynth_to_dataset
from nimare.meta.cbma.mkda import MKDAChi2
from nimare.results import MetaResult
from nimare.stats import pearson

from gradec.fetcher import (
    _fetch_dataset,
    _fetch_features,
    _fetch_metamaps,
    _fetch_model,
)
from gradec.model import _get_counts, annotate_lda
from gradec.stats import _permute_metamaps
from gradec.transform import _mni152_to_fslr
from gradec.utils import _check_ncores

N_TOP_WORDS = 3  # Number of words to show on LDA-based decoders


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

    if dset_nm == "neurosynth":
        # Download abstracts only for Neurosynth
        dset = download_abstracts(dset, "jpera054@fiu.edu")


class Decoder(metaclass=ABCMeta):
    """Base class for decoders in :mod:`~gradec.decode`."""

    def __init__(
        self,
        model_nm="lda",
        n_topics=200,
        use_fetchers=True,
        basepath=None,
        n_cores=1,
    ):
        self.model_nm = model_nm
        self.n_topics = n_topics
        self.use_fetchers = use_fetchers
        self.basepath = op.abspath(basepath) if basepath else op.abspath(".")
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

        if (self.model_nm == "term") or (self.model_nm == "lda"):
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

            metamaps_arr = gclda_model.p_voxel_g_topic_.T
            metamaps = masking.unmask(metamaps_arr, gclda_model.mask)

        return metamaps

    def fit(self, dset_nm):
        self.dset_nm = dset_nm
        if self.use_fetchers:
            # self.dset = _fetch_dataset(dset_nm, self.basepath)
            # self.model = _fetch_model(self.dset_nm, self.model_nm)
            # self.decoder = _fetch_decoder(self.dset_nm, self.model_nm)
            features = _fetch_features(self.dset_nm, self.model_nm, self.basepath)
            metamaps_fslr = _fetch_metamaps(self.dset_nm, self.model_nm, self.basepath)
            # metamaps_pmted_fslr = _fetch_nullmaps(self.dset_nm, self.model_nm)
        else:
            self.dset = _get_dataset(dset_nm, self.basepath)

            if self.model_nm == "lda":
                self.dset, self.model = annotate_lda(
                    self.dset,
                    dset_nm,
                    n_topics=self.n_topics,
                    n_cores=self.n_cores,
                )
            metamaps = self._train_decoder()
            metamaps_fslr = _mni152_to_fslr(metamaps, self.basepath)
            metamaps_pmted_fslr = _permute_metamaps(metamaps_fslr)
            features = self.get_features()

        self.maps_ = metamaps_fslr
        if self.model_nm in ["lda", "gclda"]:
            self.features_ = ["_".join(feature[:N_TOP_WORDS]) for feature in features]
        else:
            self.features_ = [feature[0] for feature in features]
        # self.null_maps_ = metamaps_pmted_fslr


class CorrelationDecoder(Decoder):
    """Decode an unthresholded image by correlating the image with meta-analytic maps."""

    def transform(self, grad_maps):
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
        # metamaps_fslr_perm_arr = _permute_metamaps(metamaps_fslr_arr)

        # Make sure we return a copy of the MetaResult
        corrs = [pearson(grad_map, self.maps_) for grad_map in grad_maps]
        data = np.array(corrs).T
        out_df = pd.DataFrame(index=self.features_, data=data)
        out_df.index.name = "feature"

        return out_df
