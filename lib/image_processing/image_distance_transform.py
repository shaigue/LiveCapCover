from typing import Tuple

import cv2
import numpy as np
from numpy import ndarray


def image_distance_transform(image: ndarray):
    # if non of the pixels are black then return a zero idt
    if not np.isin(0, image):
        return np.zeros((image.shape[0], image.shape[1]))
    idt = cv2.distanceTransform(image, cv2.DIST_L2, 3)
    return idt


def calc_image_diagonal_length(image: ndarray):
    h = image.shape[0]
    w = image.shape[1]
    return np.sqrt(h**2 + w**2)


def get_silhouette_idt_grad_from_mask(fg_mask: ndarray):
    """fg_mask is a boolean mask that should be True where it is the foreground,
     and False in background."""
    fg_mask_uint = fg_mask.astype(np.uint8) * 255
    silhouette = cv2.Laplacian(fg_mask_uint, cv2.CV_8U)
    silhouette_bool = silhouette > 0
    pre_idt = np.where(silhouette_bool, 0, 255).astype(np.uint8)
    idt = image_distance_transform(pre_idt)
    idt = (idt / idt.max()) * 255
    dx, dy = cv2.spatialGradient(idt.astype(np.uint8))
    grad = np.stack((dx, dy), axis=-1)

    return silhouette_bool, idt, grad


class NotEnoughVertices(Exception):
    pass


def sample_contour_vertices(n: int, silhouette: ndarray) -> Tuple[ndarray, ndarray]:
    indices = np.argwhere(silhouette)
    k = len(indices)
    if k < n:
        raise NotEnoughVertices
    step = k // n
    subset = [i * step for i in range(n)]
    rows, cols = indices[subset].T
    return rows, cols


