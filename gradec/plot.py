"""Plot module for gradec."""
import os

import brainspace
import matplotlib.pyplot as plt
import numpy as np
from brainspace.plotting.base import Plotter
from neuromaps.datasets import fetch_fslr
from surfplot import Plot
from surfplot.utils import threshold

from gradec.utils import get_data_dir

brainspace.OFF_SCREEN = True  # off screen rendering for examples


def plot_surf_maps(
    lh_grad,
    rh_grad,
    cmap="viridis",
    color_range=None,
    threshold_=None,
    data_dir=None,
):
    """Plot surface maps."""
    data_dir = get_data_dir(data_dir)
    neuromaps_dir = get_data_dir(os.path.join(data_dir, "neuromaps"))

    surfaces = fetch_fslr(density="32k", data_dir=neuromaps_dir)
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
    # Plotter.close_all()

    return p.build()
