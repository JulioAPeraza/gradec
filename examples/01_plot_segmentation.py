"""

.. _segmentation:


====================================================
Segmentation of the Cortical Gradient
====================================================

Gradec provides two methods for splitting the gradient spectrum.
"""

# Download HCP S1200 principal gradient
from gradec.fetcher import _fetch_gradients

gradients = _fetch_gradients(cortical=True)

###############################################################################
# 1D Segmentation
# -----------------------------------------------------------------------------
# Select and plot the principal gradient
from surfplot.utils import add_fslr_medial_wall

from gradec.plot import plot_surf_maps

principal_gradient = gradients[:, 0]

full_vertices = 64984
hemi_vertices = full_vertices // 2

prin_grad = add_fslr_medial_wall(principal_gradient)  # Add medial wall for plotting
prin_grad_lh, prin_grad_rh = prin_grad[:hemi_vertices], prin_grad[hemi_vertices:full_vertices]

fig = plot_surf_maps(prin_grad_lh, prin_grad_rh)
fig.show()

###############################################################################
# KMeans-based segmentation
# ``````````````````````````````````````````````````````````````````````````````
from gradec.segmentation import KMeansSegmentation

k_segmentation = KMeansSegmentation(n_segments=2)
k_segmentation.fit(principal_gradient)
k_grad_maps = k_segmentation.transform()

for map_i, k_grad_map in enumerate(k_grad_maps):
    k_grad_map = add_fslr_medial_wall(k_grad_map)
    k_grad_map_lh, k_grad_map_rh = (
        k_grad_map[:hemi_vertices],
        k_grad_map[hemi_vertices:full_vertices],
    )
    fig = plot_surf_maps(
        k_grad_map_lh,
        k_grad_map_rh,
        color_range=(0, 1),
        cmap="YlOrRd",
        title=f"KMeans, segment {map_i+1}",
    )
    fig.show()

###############################################################################
# PCT-based segmentation
# ``````````````````````````````````````````````````````````````````````````````
from gradec.segmentation import PCTSegmentation

p_segmentation = PCTSegmentation(n_segments=2)
p_segmentation.fit(principal_gradient)
p_grad_maps = p_segmentation.transform()

for map_i, p_grad_map in enumerate(p_grad_maps):
    p_grad_map = add_fslr_medial_wall(p_grad_map)
    p_grad_map_lh, p_grad_map_rh = (
        p_grad_map[:hemi_vertices],
        p_grad_map[hemi_vertices:full_vertices],
    )
    fig = plot_surf_maps(
        p_grad_map_lh,
        p_grad_map_rh,
        color_range=(0, 1),
        cmap="YlOrRd",
        title=f"PCT, segment {map_i+1}",
    )
    fig.show()


###############################################################################
# Multidimensional Segmentation
# -----------------------------------------------------------------------------
# Select the first four gradients
gradients_4d = gradients[:, :4]

###############################################################################
# KMeans-based segmentation
# ``````````````````````````````````````````````````````````````````````````````
from gradec.segmentation import KMeansSegmentation

k4_segmentation = KMeansSegmentation(n_segments=4)
k4_segmentation.fit(gradients_4d)
k4_grad_maps = k4_segmentation.transform()

for map_i, k4_grad_map in enumerate(k4_grad_maps):
    k4_grad_map = add_fslr_medial_wall(k4_grad_map)
    k4_grad_map_lh, k4_grad_map_rh = (
        k4_grad_map[:hemi_vertices],
        k4_grad_map[hemi_vertices:full_vertices],
    )
    fig = plot_surf_maps(
        k4_grad_map_lh,
        k4_grad_map_rh,
        color_range=(0, 1),
        cmap="YlOrRd",
        title=f"KMeans, cluster {map_i+1}",
    )
    fig.show()
