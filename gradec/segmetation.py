"""Segmentation module."""
from abc import ABCMeta

import numpy as np
from sklearn.cluster import KMeans

class Segmentation(metaclass=ABCMeta):
    """Base class for segmentation methods."""

    def __init__(self):
        pass

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
    
    def transform(self, gradient):
        """Transform gradient to maps."""
        segments = self.segmentation_dict["segments"]
        peaks = self.segmentation_dict["peaks"]

        grad_segments = []
        for seg_i, segment in enumerate(segments):
            grad_maps = []
            for map_i, grad_map in enumerate(segment):
                # Vertices located above the cluster_centers_ in the segment map
                # were translated relative to the maximum
                grad_map_peak = peaks[seg_i][map_i]
                vrtxs_non_zero = np.where(grad_map != 0)
                vrtxs_to_translate = np.where((grad_map > grad_map_peak) & (grad_map != 0))
                grad_map[vrtxs_to_translate] = grad_map_peak - np.abs(grad_map[vrtxs_to_translate])

                # Translate relative to zero
                grad_map_min = grad_map[vrtxs_non_zero].min()
                grad_map[vrtxs_non_zero] = np.abs(grad_map_min) + grad_map[vrtxs_non_zero]

                grad_maps.append(grad_map)
            grad_segments.append(grad_maps)

        return grad_segments

class PCTLSegmentation(Segmentation):
    """Percentile-based segmentation.

    Thi method implements the original method described in (Margulies et al., 2016)
    in which the whole-brain gradient is segmented into equidistant gradient segments.

    Parameters
    ----------
    gradient : numpy.ndarray
        Gradient vector.
    n_segments : int
        Total number of segments.
    min_n_segments : int
        Minimum number of segments.
    """

    def __init__(
        self,
        segmentation_fn,
        n_segments,
        min_n_segments=3,
    ):
        self.segmentation_fn = segmentation_fn
        self.n_segments = n_segments
        self.min_n_segments = min_n_segments

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
            Vertices labeled.
        labels :
        boundaries :
        peaks :
        """
        segments = []
        labels = []
        boundaries = []
        peaks = []
        for n_segment in range(self.min_n_segments, self.n_segments + self.min_n_segments):
            step = 100 / n_segment
            bins = list(
                zip(np.linspace(0, 100 - step, n_segment), np.linspace(step, 100, n_segment))
            )

            gradient_maps = []
            labels_arr = np.zeros_like(gradient)
            map_peaks = []
            map_bounds = [gradient.min()]
            for i, bin in enumerate(bins):
                # Get boundary points
                min_, max_ = np.percentile(gradient[np.where(gradient)], bin)

                # Threshold gradient map based on bin, but don’t binarize.
                thresh_arr = gradient.copy()
                thresh_arr[thresh_arr < min_] = 0
                thresh_arr[thresh_arr > max_] = 0
                gradient_maps.append(thresh_arr)

                non_zero_arr = np.where(thresh_arr != 0)
                labels_arr[non_zero_arr] = i
                map_bounds.append(max_)
                # Peak activation = median 50th percentile of the segment
                map_peaks.append(np.median(thresh_arr[non_zero_arr]))

            segments.append(gradient_maps)
            labels.append(labels_arr)
            boundaries.append(map_bounds)
            peaks.append(map_peaks)

        return segments, labels, boundaries, peaks


class KMeansSegmentation(Segmentation):
    """KMeans-based segmentation.

    This method relied on 1D k-means clustering, which has previously
    been used to define clusters of functional connectivity matrices
    to establish a brain-wide parcellation.

    Parameters
    ----------
    gradient : numpy.ndarray
        Gradient vector.
    n_segments : int
        Total number of segments.
    min_n_segments : int
        Minimum number of segments.
    """

    def __init__(
        self,
        segmentation_fn,
        n_segments,
        min_n_segments=3,
    ):
        self.segmentation_fn = segmentation_fn
        self.n_segments = n_segments
        self.min_n_segments = min_n_segments

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
        segments = []
        labels = []
        boundaries = []
        peaks = []
        for n_segment in range(self.min_n_segments, self.n_segments + self.min_n_segments):
            kmeans_model = KMeans(
                n_clusters=n_segment,
                init="k-means++",
                n_init=10,
                random_state=0,
                algorithm="elkan",
            ).fit(gradient.reshape(-1, 1))

            # Get order mapper from map_peaks
            map_peaks = kmeans_model.cluster_centers_.flatten()
            order_idx = np.argsort(map_peaks)
            order_mapper = np.zeros_like(order_idx)
            order_mapper[order_idx] = np.arange(n_segment)

            # Reorder labels based on map_peaks order
            labels_arr = order_mapper[kmeans_model.labels_]

            gradient_maps = []
            map_bounds = [gradient.min()]
            for i in range(n_segment):
                map_arr = np.zeros_like(gradient)
                map_arr[labels_arr == i] = gradient[labels_arr == i]
                gradient_maps.append(map_arr)

                map_bounds.append(gradient[labels_arr == i].max())

            segments.append(gradient_maps)
            labels.append(labels_arr)
            boundaries.append(map_bounds)
            peaks.append(np.sort(map_peaks))

        return segments, labels, boundaries, peaks
