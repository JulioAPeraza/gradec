"""

.. _segmentation:

====================================================
Segmentation of the Cortical Gradient
====================================================

Gradec provides two methods for splitting the gradient spectrum.
"""
import matplotlib.pyplot as plt

from gradec.fetcher import _fetch_principal_gradients

###############################################################################
# Download HCP S1200 principal gradient
# -----------------------------------------------------------------------------

principal_gradient = _fetch_principal_gradients()

from brainspace.plotting.base import Plotter

###############################################################################
# Plot principal gradient
# ``````````````````````````````````````````````````````````````````````````````
from surfplot.utils import add_fslr_medial_wall

from gradec.plot import plot_surf_maps

full_vertices = 64984
hemi_vertices = full_vertices // 2

prin_grad = add_fslr_medial_wall(principal_gradient)  # Add medial wall for plotting
prin_grad_lh, prin_grad_rh = prin_grad[:hemi_vertices], prin_grad[hemi_vertices:full_vertices]

p_obj = plot_surf_maps(prin_grad_lh, prin_grad_rh)
fig = p_obj.build()
Plotter.close_all()
fig.show()

###############################################################################
# Segment the principal gradient
# ``````````````````````````````````````````````````````````````````````````````
from gradec.segmentation import KMeansSegmentation

segmentation = KMeansSegmentation(5)
segmentation.fit(principal_gradient)
grad_maps = segmentation.transform()

###############################################################################
# Plot gradients maps
# ``````````````````````````````````````````````````````````````````````````````
for grad_map in grad_maps:
    grad_map = add_fslr_medial_wall(grad_map)
    grad_map_lh, grad_map_rh = grad_map[:hemi_vertices], grad_map[hemi_vertices:full_vertices]
    p_obj = plot_surf_maps(grad_map_lh, grad_map_rh)
    fig = p_obj.build()
    Plotter.close_all()
    fig.show()
