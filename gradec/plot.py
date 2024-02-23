"""Plot module for gradec."""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from neuromaps.datasets import fetch_civet, fetch_fsaverage, fetch_fslr
from surfplot import Plot
from surfplot.utils import threshold
from wordcloud import WordCloud

from gradec.fetcher import _fetch_model
from gradec.utils import get_data_dir


def _get_twfrequencies(dset_nm, model_nm, n_top_terms, data_dir=None):
    """Get top word frequencies from a topic model."""
    model_obj = _fetch_model(dset_nm, model_nm, data_dir=data_dir)

    topic_word_weights = (
        model_obj.p_word_g_topic_.T
        if model_nm == "gclda"
        else model_obj.distributions_["p_topic_g_word"]
    )

    n_topics = topic_word_weights.shape[0]
    sorted_weights_idxs = np.argsort(-topic_word_weights, axis=1)
    frequencies_lst = []
    for topic_i in range(n_topics):
        frequencies = topic_word_weights[topic_i, sorted_weights_idxs[topic_i, :]][
            :n_top_terms
        ].tolist()
        frequencies = [freq / np.max(frequencies) for freq in frequencies]
        frequencies = np.round(frequencies, 3).tolist()
        frequencies_lst.append(frequencies)

    return frequencies_lst


def plot_radar(
    corrs,
    features,
    model_nm,
    cmap="YlOrRd",
    n_top_terms=3,
    fig=None,
    ax=None,
    out_fig=None,
):
    """Plot radar chart."""
    n_rows = min(len(corrs), 10)
    angle_zero = 0
    fontsize = 36

    # Sort features and correlations
    corrs = np.array(corrs)
    sorted_indices = np.argsort(-corrs)
    corrs = corrs[sorted_indices]
    features = np.array(features, dtype=object)[sorted_indices]

    corrs = corrs[:n_rows]
    features = features[:n_rows]
    angles = [(angle_zero + (n / float(n_rows) * 2 * np.pi)) for n in range(n_rows)]
    if model_nm in ["lda", "gclda"]:
        features = ["\n".join(feature[:n_top_terms]).replace(" ", "\n") for feature in features]
    else:
        features = [feature.replace(" ", "\n") for feature in features]

    roundup_corr = math.ceil(corrs.max() * 10) / 10

    # Define color scheme
    plt.rcParams["text.color"] = "#1f1f1f"
    cmap_ = colormaps.get_cmap(cmap)
    norm = plt.Normalize(vmin=corrs.min(), vmax=corrs.max())
    colors = cmap_(norm(corrs))

    # Plot radar
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": "polar"})

    ax.set_theta_offset(0)
    ax.set_ylim(-0.1, roundup_corr)

    ax.bar(angles, corrs, color=colors, alpha=0.9, width=0.52, zorder=10)
    ax.vlines(angles, 0, roundup_corr, color="grey", ls=(0, (4, 4)), zorder=11)

    ax.set_xticks(angles)
    ax.set_xticklabels(features, size=fontsize, zorder=13)

    ax.xaxis.grid(False)

    step = 0.1 + 1e-09
    yticks = np.round(np.arange(0, roundup_corr + step, step), 1)
    ax.set_yticklabels([])
    ax.set_yticks(yticks)

    ax.spines["start"].set_color("none")
    ax.spines["polar"].set_color("none")

    xticks = ax.xaxis.get_major_ticks()
    [xtick.set_pad(90) for xtick in xticks]

    sep = 0.06
    [
        ax.text(
            np.pi / 2,
            ytick - sep,
            f"{ytick}",
            ha="center",
            size=fontsize - 2,
            color="grey",
            zorder=12,
        )
        for ytick in yticks
    ]

    if out_fig is None:
        return fig

    fig.savefig(out_fig, bbox_inches="tight")
    plt.close()


def plot_cloud(
    corrs,
    features,
    model_nm,
    frequencies=None,
    cmap="YlOrRd",
    n_top_terms=3,
    width=9,
    height=5,
    dpi=100,
    fig=None,
    ax=None,
    out_fig=None,
):
    """Plot word cloud."""
    frequencies_dict = {}
    if model_nm in ["lda", "gclda"]:
        features = [feature[:n_top_terms] for feature in features]
        frequencies = [frequencie[:n_top_terms] for frequencie in frequencies]

        for corr, features, frequency in zip(corrs, features, frequencies):
            for word, freq in zip(features, frequency):
                if word not in frequencies_dict:
                    frequencies_dict[word] = freq * corr
    else:
        for word, corr in zip(features, corrs):
            if word not in frequencies_dict:
                frequencies_dict[word] = corr

    if fig is None:
        if ax is None:
            fig, ax = plt.subplots(figsize=(width, height))
        else:
            fig, _ = plt.subplots(figsize=(width, height))

    wc = WordCloud(
        width=width * dpi,
        height=height * dpi,
        background_color="white",
        random_state=0,
        colormap=cmap,
    )
    wc.generate_from_frequencies(frequencies=frequencies_dict)
    ax.imshow(wc)

    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if out_fig is None:
        return fig

    fig.savefig(out_fig, bbox_inches="tight")
    plt.close()


def plot_surf_maps(
    lh_grad,
    rh_grad,
    space="fsLR",
    density="32k",
    cmap="viridis",
    title=None,
    color_range=None,
    threshold_=None,
    data_dir=None,
    out_fig=None,
):
    """Plot surface maps."""
    data_dir = get_data_dir(data_dir)
    neuromaps_dir = get_data_dir(os.path.join(data_dir, "neuromaps"))

    if space == "fsLR":
        surfaces = fetch_fslr(density=density, data_dir=neuromaps_dir)
    elif space == "fsaverage":
        surfaces = fetch_fsaverage(density=density, data_dir=neuromaps_dir)
    elif space == "civet":
        surfaces = fetch_civet(density=density, data_dir=neuromaps_dir)

    lh, rh = surfaces["inflated"]
    sulc_lh, sulc_rh = surfaces["sulc"]

    if threshold_:
        lh_grad = threshold(lh_grad, threshold_)
        rh_grad = threshold(rh_grad, threshold_)

    p = Plot(surf_lh=lh, surf_rh=rh, layout="grid")
    p.add_layer({"left": sulc_lh, "right": sulc_rh}, cmap="binary_r", cbar=False)
    p.add_layer(
        {"left": lh_grad, "right": rh_grad},
        cmap=cmap,
        cbar=True,
        color_range=color_range,
    )

    fig = p.build()

    if title is not None:
        fig.axes[0].set_title(title, pad=-3)

    if out_fig is None:
        return fig

    fig.savefig(out_fig, bbox_inches="tight")
    plt.close()
