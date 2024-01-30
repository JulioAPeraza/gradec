"""Segmentation module."""

from abc import ABCMeta

import numpy as np
from sklearn.cluster import KMeans


class Segmentation(metaclass=ABCMeta):
    """Base class for segmentation methods.

    Parameters
    ----------
    n_segments : int
        Total number of segments.
    """

    def __init__(self, n_segments=5):
        self.n_segments = n_segments

    def fit(self, gradient):
        """Fit Segmentation to gradient."""
        segments, labels, boundaries, peaks = self._fit(gradient)

        # Save results to dictionary file
        self.segmentation_dict = {
            "segments": segments,
            "labels": labels,
            "boundaries": boundaries,
            "peaks": peaks,
        }

        return self

    def transform(self):
        """Transform gradient to maps."""
        segments = self.segmentation_dict["segments"]
        peaks = self.segmentation_dict["peaks"]

        grad_maps = []
        for map_i, grad_map in enumerate(segments):
            # Vertices located above the cluster_centers_ in the segment map
            # were translated relative to the maximum
            grad_map_peak = peaks[map_i]
            vrtxs_non_zero = np.where(grad_map != 0)
            vrtxs_to_translate = np.where((grad_map > grad_map_peak) & (grad_map != 0))
            grad_map[vrtxs_to_translate] = grad_map_peak - np.abs(grad_map[vrtxs_to_translate])

            # Translate relative to zero
            grad_map_min = grad_map[vrtxs_non_zero].min()
            grad_map[vrtxs_non_zero] = np.abs(grad_map_min) + grad_map[vrtxs_non_zero]

            grad_maps.append(grad_map)

        return grad_maps


class PCTSegmentation(Segmentation):
    """Percentile-based segmentation.

    This method implements the original method described in (Margulies et al., 2016)
    in which the whole-brain gradient is segmented into equidistant gradient segments.
    """

    def _fit(self, gradient):
        """Fit Segmentation to gradient.

        Parameters
        ----------
        gradient : numpy.ndarray
            Gradient vector.

        Returns
        -------
        segments : list of numpy.ndarray
            List with thresholded gradients maps.
        labels :
        boundaries :
        peaks :
        """
        step = 100 / self.n_segments
        bins = [(i, i + step) for i in range(0, int(100 - step + 1), int(step))]

        segments = []
        labels = []
        labels = np.zeros_like(gradient)
        boundaries = [gradient.min()]
        peaks = []
        for i, bin in enumerate(bins):
            # Get boundary points
            min_, max_ = np.percentile(gradient[np.where(gradient)], bin)

            # Threshold gradient map based on bin, but don’t binarize.
            thresh_arr = gradient.copy()
            thresh_arr[thresh_arr < min_] = 0
            thresh_arr[thresh_arr > max_] = 0
            segments.append(thresh_arr)

            non_zero_arr = np.where(thresh_arr != 0)
            labels[non_zero_arr] = i
            boundaries.append(max_)
            # Peak activation = median 50th percentile of the segment
            peaks.append(np.median(thresh_arr[non_zero_arr]))

        return segments, labels, boundaries, peaks


class KMeansSegmentation(Segmentation):
    """KMeans-based segmentation.

    This method relies on 1D k-means clustering, which has previously
    been used to define clusters of functional connectivity matrices
    to establish a brain-wide parcellation.
    """

    def _fit(self, gradient):
        """Fit Segmentation to gradient.

        Parameters
        ----------
        gradient : numpy.ndarray
            Gradient vector.

        Returns
        -------
        segments : list of numpy.ndarray
            List with thresholded gradients maps.
        kde_labels : list of numpy.ndarray
            Vertices labeled
        labels :
        boundaries :
        peaks :
        """
        kmeans_model = KMeans(
            n_clusters=self.n_segments,
            init="k-means++",
            n_init=10,
            random_state=0,
            algorithm="elkan",
        ).fit(gradient.reshape(-1, 1))

        # Get order mapper from map_peaks
        map_peaks = kmeans_model.cluster_centers_.flatten()
        peaks = np.sort(map_peaks)
        order_idx = np.argsort(map_peaks)
        order_mapper = np.zeros_like(order_idx)
        order_mapper[order_idx] = np.arange(self.n_segments)

        # Reorder labels based on map_peaks order
        labels = order_mapper[kmeans_model.labels_]

        segments = []
        boundaries = [gradient.min()]
        for i in range(self.n_segments):
            map_arr = np.zeros_like(gradient)
            map_arr[labels == i] = gradient[labels == i]
            segments.append(map_arr)

            boundaries.append(gradient[labels == i].max())

        return segments, labels, boundaries, peaks
