"""Models for Gradec."""

import os.path as op

import numpy as np
import pandas as pd
from nimare.annotate.text import generate_counts
from nimare.base import NiMAREBase
from nimare.utils import _check_ncores, get_resource_path
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from gradec.fetcher import _fetch_neuroquery_counts


def _generate_counts(
    text_df,
    vocabulary=None,
    text_column="abstract",
    tfidf=True,
    min_df=0.01,
    max_df=0.99,
):
    """Generate tf-idf/counts weights for unigrams/bigrams derived from textual data.

    Parameters
    ----------
    text_df : (D x 2) :obj:`pandas.DataFrame`
        A DataFrame with two columns ('id' and 'text'). D = document.

    Returns
    -------
    weights_df : (D x T) :obj:`pandas.DataFrame`
        A DataFrame where the index is 'id' and the columns are the
        unigrams/bigrams derived from the data. D = document. T = term.
    """
    if text_column not in text_df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")

    # Remove rows with empty text cells
    orig_ids = text_df["id"].tolist()
    text_df = text_df.fillna("")
    keep_ids = text_df.loc[text_df[text_column] != "", "id"]
    text_df = text_df.loc[text_df["id"].isin(keep_ids)]

    if len(keep_ids) != len(orig_ids):
        print(f"\t\tRetaining {len(keep_ids)}/{len(orig_ids)} studies", flush=True)

    ids = text_df["id"].tolist()
    text = text_df[text_column].tolist()
    stoplist = op.join(get_resource_path(), "neurosynth_stoplist.txt")
    with open(stoplist, "r") as fo:
        stop_words = fo.read().splitlines()

    if tfidf:
        vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            ngram_range=(1, 2),
            vocabulary=vocabulary,
            stop_words=stop_words,
        )
    else:
        vectorizer = CountVectorizer(
            min_df=min_df,
            max_df=max_df,
            ngram_range=(1, 2),
            vocabulary=vocabulary,
            stop_words=stop_words,
        )
    weights = vectorizer.fit_transform(text).toarray()

    names = vectorizer.get_feature_names_out()
    names = [str(name) for name in names]
    weights_df = pd.DataFrame(weights, columns=names, index=ids)
    weights_df.index.name = "id"
    return weights_df


def _get_counts(dataset, dataset_nm, feature_group):
    """Get counts weights for unigrams/bigrams derived from textual data."""
    if dataset_nm == "neurosynth":
        feature_names = dataset.annotations.columns.values
        feature_names = [f for f in feature_names if f.startswith(feature_group)]
        vocabulary = [f.split("__")[-1] for f in feature_names]
        counts_df = _generate_counts(
            dataset.texts,
            vocabulary=vocabulary,
            text_column="abstract",
            tfidf=False,
            max_df=len(dataset.ids) - 2,
            min_df=2,
        )

    elif dataset_nm == "neuroquery":
        counts_arr = _fetch_neuroquery_counts()

        # Generate the IDs from original id list (without sorting)
        ids = dataset.annotations.sort_index()["id"].tolist()
        feature_names = dataset.annotations.columns.values
        feature_names = [f for f in feature_names if f.startswith(feature_group)]
        vocabulary = [f.split("__")[-1] for f in feature_names]

        counts_df = pd.DataFrame(counts_arr, columns=vocabulary, index=ids)
        counts_df.index.name = "id"

        # Sorting by id to match the sorting perform in NiMARE Dataset
        counts_df = counts_df.sort_index()

    return counts_df


class LDAModel(NiMAREBase):
    """Generate a latent Dirichlet allocation (LDA) topic model.

    This class is a light wrapper around scikit-learn tools for tokenization and LDA.

    Parameters
    ----------
    n_topics : :obj:`int`
        Number of topics for topic model. This corresponds to the model's ``n_components``
        parameter. Must be an integer >= 1.
    max_iter : :obj:`int`, optional
        Maximum number of iterations to use during model fitting. Default = 1000.
    alpha : :obj:`float` or None, optional
        The ``alpha`` value for the model. This corresponds to the model's ``doc_topic_prior``
        parameter. Default is None, which evaluates to ``1 / n_topics``,
        as was used in :footcite:t:`poldrack2012discovering`.
    beta : :obj:`float` or None, optional
        The ``beta`` value for the model. This corresponds to the model's ``topic_word_prior``
        parameter. If None, it evaluates to ``1 / n_topics``.
        Default is 0.001, which was used in :footcite:t:`poldrack2012discovering`.
    text_column : :obj:`str`, optional
        The source of text to use for the model. This should correspond to an existing column
        in the :py:attr:`~nimare.dataset.Dataset.texts` attribute. Default is "abstract".
    n_cores : :obj:`int`, optional
        Number of cores to use for parallelization.
        If <=0, defaults to using all available cores.
        Default is 1.

    Attributes
    ----------
    model : :obj:`~sklearn.decomposition.LatentDirichletAllocation`

    Notes
    -----
    Adapted from: https://github.com/neurostuff/NiMARE/blob/main/nimare/annotate/lda.py.

    Latent Dirichlet allocation was first developed in :footcite:t:`blei2003latent`,
    and was first applied to neuroimaging articles in :footcite:t:`poldrack2012discovering`.

    References
    ----------
    .. footbibliography::

    See Also
    --------
    :class:`~sklearn.feature_extraction.text.CountVectorizer`: Used to build a vocabulary of terms
        and their associated counts from texts in the ``self.text_column`` of the Dataset's
        ``texts`` attribute.
    :class:`~sklearn.decomposition.LatentDirichletAllocation`: Used to train the LDA model.
    """

    def __init__(
        self, n_topics, max_iter=1000, alpha=None, beta=0.001, text_column="abstract", n_cores=1
    ):
        self.n_topics = n_topics
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.text_column = text_column
        self.n_cores = _check_ncores(n_cores)

        self.model = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=max_iter,
            learning_method="batch",
            doc_topic_prior=alpha,
            topic_word_prior=beta,
            n_jobs=n_cores,
        )

    def fit(self, dset, counts_df=None):
        """Fit the LDA topic model to text from a Dataset.

        Parameters
        ----------
        dset : :obj:`~nimare.dataset.Dataset`
            A Dataset with, at minimum, text available in the ``self.text_column`` column of its
            :py:attr:`~nimare.dataset.Dataset.texts` attribute.
        count_df : :obj:`pandas.DataFrame`
            A DataFrame with feature counts for the model. The index is 'id',
            used for identifying studies. Other columns are features (e.g.,
            unigrams and bigrams from Neurosynth), where each value is the number
            of times the feature is found in a given article.

        Returns
        -------
        dset : :obj:`~nimare.dataset.Dataset`
            A new Dataset with an updated :py:attr:`~nimare.dataset.Dataset.annotations` attribute.

        Attributes
        ----------
        distributions_ : :obj:`dict`
            A dictionary containing additional distributions produced by the model, including:

                -   ``p_topic_g_word``: :obj:`numpy.ndarray` of shape (n_topics, n_tokens)
                    containing the topic-term weights for the model.
                -   ``p_topic_g_word_df``: :obj:`pandas.DataFrame` of shape (n_topics, n_tokens)
                    containing the topic-term weights for the model.
        """
        if counts_df is None:
            counts_df = generate_counts(
                dset.texts,
                text_column=self.text_column,
                tfidf=False,
                max_df=len(dset.ids) - 2,
                min_df=2,
            )

        vocabulary = counts_df.columns.to_numpy()
        count_values = counts_df.values
        study_ids = counts_df.index.tolist()

        doc_topic_weights = self.model.fit_transform(count_values)
        topic_word_weights = self.model.components_

        # Get top 3 words for each topic for annotation
        sorted_weights_idxs = np.argsort(-topic_word_weights, axis=1)
        top_tokens = [
            "_".join(vocabulary[sorted_weights_idxs[topic_i, :]][:3])
            for topic_i in range(self.n_topics)
        ]
        topic_names = [
            f"LDA{self.n_topics}__{i + 1}_{top_tokens[i]}" for i in range(self.n_topics)
        ]

        doc_topic_weights_df = pd.DataFrame(
            index=study_ids,
            columns=topic_names,
            data=doc_topic_weights,
        )
        topic_word_weights_df = pd.DataFrame(
            index=topic_names,
            columns=vocabulary,
            data=topic_word_weights,
        )
        self.distributions_ = {
            "p_topic_g_word": topic_word_weights,
            "p_topic_g_word_df": topic_word_weights_df,
        }

        annotations = dset.annotations.copy()
        annotations = pd.merge(annotations, doc_topic_weights_df, left_on="id", right_index=True)
        new_dset = dset.copy()
        new_dset.annotations = annotations
        return new_dset


def annotate_lda(dataset, dataset_nm, feature_group, n_topics=200, n_cores=1):
    """Annotate Dataset with the resutls of an LDA model.

    Parameters
    ----------
    dset : :obj:`~nimare.dataset.Dataset`
        A Dataset with, at minimum, text available in the ``self.text_column`` column of its
        :py:attr:`~nimare.dataset.Dataset.texts` attribute.
    n_topics : :obj:`int`
        Number of topics for topic model. This corresponds to the model's ``n_components``
        parameter. Must be an integer >= 1.
    dset_name: str
        Dataset name. Possible options: "neurosynth" or "neuroquery"
    data_dir: str
        Path to data directory.
    n_cores : :obj:`int`, optional
        Number of cores to use for parallelization.
        If <=0, defaults to using all available cores.
        Default is 1.

    Returns
    -------
    dset : :obj:`~nimare.dataset.Dataset`
        A new Dataset with an updated :py:attr:`~nimare.dataset.Dataset.annotations` attribute.
    """
    counts_df = _get_counts(dataset, dataset_nm, feature_group)

    model = LDAModel(n_topics=n_topics, max_iter=1000, n_cores=n_cores)
    dataset = model.fit(dataset, counts_df)

    return dataset, model
