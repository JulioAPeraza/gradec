"""Segmentation module."""

from abc import ABCMeta

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

SIGMA_0 = 1


class Segmentation(metaclass=ABCMeta):
    """Base class for segmentation methods.

    Parameters
    ----------
    n_segments : int
        Total number of segments.

    Attributes
    ----------
    multidimensional : bool
        Whether the input is multidimensional.
    """

    def __init__(self, n_segments=5):
        self.n_segments = n_segments
        self.multidimensional = False

    def fit(self, gradients):
        """Fit Segmentation to gradients."""

    def transform(self):
        """Transform gradients to maps."""
        grad_maps = []
        for segment_i, (segment, peak) in enumerate(zip(self.segments, self.peaks)):
            mask = self.labels == segment_i
            segment_val = segment[mask]

            distances = pairwise_distances(
                segment_val, peak.reshape(1, -1), metric="euclidean"
            ).flatten()

            mean_dist = np.mean(distances)
            sigma = mean_dist * SIGMA_0
            affinity = np.exp(-(distances**2) / (2 * sigma**2))

            pseudo_act_map = np.zeros(segment.shape[0], dtype=segment.dtype)
            pseudo_act_map[mask] = np.array(affinity)

            grad_maps.append(pseudo_act_map)

        return grad_maps


class PCTSegmentation(Segmentation):
    """Percentile-based segmentation.

    This method implements the original method described in (Margulies et al., 2016)
    in which the whole-brain gradients is segmented into equidistant gradients segments.
    """

    def fit(self, gradients):
        """Fit Segmentation to gradient.

        Parameters
        ----------
        gradients : numpy.ndarray
            Gradients vector.

        Attributes
        ----------
        segments : list of numpy.ndarray
            List with thresholded gradients maps.
        labels :
            L
        boundaries :

        peaks :
        """
        if gradients.ndim > 1:
            raise ValueError("PCTSegmentation only supports 1D gradients.")
        else:
            # Reshape gradients to 2D to avoid issues with pairwise_distances in transform
            gradients = gradients.reshape(-1, 1)

        step = 100 / self.n_segments
        bins = [(i, i + step) for i in range(0, int(100 - step + 1), int(step))]

        segments = []
        labels = []
        labels = np.zeros(gradients.shape[0], dtype=int)
        boundaries = [gradients.min()]
        peaks = []
        for i, bin in enumerate(bins):
            # Get boundary points
            min_, max_ = np.percentile(gradients[np.where(gradients)], bin)

            # Threshold gradients map based on bin, but don’t binarize.
            mask = (gradients >= min_) & (gradients <= max_)
            mask = mask.flatten()

            thresh_arr = gradients.copy()
            thresh_arr[~mask] = 0
            segments.append(thresh_arr)

            labels[mask] = i
            boundaries.append(max_)
            # Peak activation = median 50th percentile of the segment
            peaks.append(np.median(thresh_arr[mask]))

        # Replace the first and last peaks with the min and max of the gradients
        peaks[0], peaks[-1] = gradients.min(), gradients.max()

        self.segments = segments
        self.labels = labels
        self.boundaries = boundaries
        self.peaks = peaks

        return self


class KMeansSegmentation(Segmentation):
    """KMeans-based segmentation.

    This method relies on 1D k-means clustering, which has previously
    been used to define clusters of functional connectivity matrices
    to establish a brain-wide parcellation.
    """

    def fit(self, gradients):
        """Fit Segmentation to gradients.

        Parameters
        ----------
        gradients : (V x D) numpy.ndarray
            Gradients vector, where V is the number of vertices and D is the number of dimensions.

        Attributes
        ----------
        segments : list of numpy.ndarray
            List with thresholded gradients maps.
        labels :
        boundaries :
        peaks :
        """
        if gradients.ndim > 1 and gradients.shape[1] > 1:
            self.multidimensional = True
        else:
            gradients = gradients.reshape(-1, 1)

        kmeans_model = KMeans(
            n_clusters=self.n_segments,
            init="k-means++",
            n_init=10,
            random_state=0,
            algorithm="elkan",
        ).fit(gradients)

        # Get order mapper from map_peaks
        peaks_unsrt = kmeans_model.cluster_centers_
        map_peaks = peaks_unsrt[:, 0].flatten()  # Order base on principal gradient
        order_idx = np.argsort(map_peaks)
        order_mapper = np.zeros_like(order_idx)
        order_mapper[order_idx] = np.arange(self.n_segments)

        # Reorder labels based on map_peaks order
        labels = order_mapper[kmeans_model.labels_]
        peaks = peaks_unsrt[order_idx, :]

        # Save the boundaries of the segments only for 1D case
        boundaries = [] if self.multidimensional else [gradients.min()]

        segments = []
        for i in range(self.n_segments):
            map_arr = np.zeros_like(gradients)
            map_arr[labels == i] = gradients[labels == i]
            segments.append(map_arr)

            if not self.multidimensional:
                boundaries.append(gradients[labels == i].max())

        if not self.multidimensional:
            # Replace the first and last peaks with the min and max of the gradients for 1D
            peaks[0], peaks[-1] = gradients.min(), gradients.max()

        self.segments = segments
        self.labels = labels
        self.boundaries = boundaries
        self.peaks = peaks

        return self
