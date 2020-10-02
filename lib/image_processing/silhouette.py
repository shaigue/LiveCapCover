import os
from pathlib import Path

import cv2
import numpy as np
from numpy import ndarray

from PIL import Image


class SilhouetteExtractor:
    def __init__(self):
        self.substructor = cv2.createBackgroundSubtractorMOG2(varThreshold=100)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def process(self, frame: ndarray, drop_weak: bool = False):
        if not isinstance(frame, ndarray):
            raise TypeError('frame should be ndarray, got ', type(frame))
        if len(frame.shape) != 2:
            raise ValueError('image should be in grayscale, (H,W), got ', frame.shape)

        mask = self.substructor.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, self.kernel)
        if drop_weak:
            mask = np.where(mask == 255, np.zeros_like(mask), np.full_like(mask, 255))
        else:
            mask = np.where(mask == 0, np.full_like(mask, 255), np.zeros_like(mask))
        return mask
            # cv2.distanceTransform(mask, cv2.DIST_L2, 3)


class ColorSilhouetteExtractor:
    def __init__(self, tolerance: int = 10):
        """
        :param tolerance: how much shift we allow to the background colors
        """
        self.tolerance = tolerance
        self.colors = []

    def fit(self, ds, step: int = 30, max_colors: int = 5, min_count: int = 5):
        # find the colors that don't change for the entire scene
        n_frames = len(ds)
        # it is important to use signed types so that we can compare negative values as well
        first_frame = ds[0]['frame'].astype(np.int32)
        # holds what pixels did not change
        total_mask = np.ones(first_frame.shape[:2], dtype=np.bool)
        for i in range(1, n_frames, step):
            frame = ds[i]['frame'].astype(np.int32)
            diff = frame - first_frame
            # each color channel separately, and then and them all, and with total_mask
            current_mask = (diff < self.tolerance) & (diff > -self.tolerance)
            current_mask = np.all(current_mask, axis=2)
            total_mask &= current_mask
        # plt.imshow(total_mask)
        # plt.show()
        colors, counts = np.unique(first_frame[total_mask], axis=0, return_counts=True)
        # filter the colors that are less frequent from 'min_count'
        subset = counts >= min_count
        colors = colors[subset]
        counts = counts[subset]

        while len(colors) > 0 and len(self.colors) < max_colors:
            max_idx = counts.argmax()
            color = colors[max_idx]
            self.colors.append(color)
            # remove the similar colors from the colors
            diff = colors - color
            similar = (diff < self.tolerance) & (diff > -self.tolerance)
            similar = np.all(similar, axis=1)
            colors = colors[~similar]
            counts = counts[~similar]

    def process(self, frame: ndarray):
        # filter out the similar colors, and return the mask
        mask = np.zeros(frame.shape[:2], dtype=np.bool)
        frame = frame.astype(np.int32)
        for color in self.colors:
            diff = frame - color
            similar = (diff < self.tolerance) & (diff > -self.tolerance)
            similar = np.all(similar, axis=2)
            mask |= similar
            # plt.imshow(similar)
            # plt.show()
        return mask.astype(np.uint8) * 255


class LivecapBGS:
    def __init__(self, bg_image_dir: Path):
        self.bgs = cv2.createBackgroundSubtractorMOG2(varThreshold=150)
        for image in bg_image_dir.glob('*'):
            image = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
            self.bgs.apply(image)

    def process(self, frame: ndarray):
        mask = self.bgs.apply(frame, learningRate=0)
        return mask
