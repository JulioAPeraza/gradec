"""Plot module for gradec."""
import os

from neuromaps.datasets import fetch_civet, fetch_fsaverage, fetch_fslr
from surfplot import Plot
from surfplot.utils import threshold

from gradec.utils import get_data_dir


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

    return fig
