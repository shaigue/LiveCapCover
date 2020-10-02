from dataclasses import dataclass
from pathlib import Path
import pickle
import os
from typing import Union, List

import cv2
from numpy import ndarray
import numpy as np
from PIL import Image

from lib.image_processing.vibe_transformations import kp_to_orig_image
from lib.utils.utils import read_image_as_numpy, read_rgb_image_as_numpy


@dataclass
class VibeEntry:
    theta: ndarray
    verts: ndarray
    kp_2d: ndarray
    kp_3d: ndarray
    rotmat: ndarray


@dataclass
class LiveCapEntry:
    frame: ndarray = None
    bbox: ndarray = None
    vibe: VibeEntry = None
    silhouette: ndarray = None


def convert_kp_2d_to_pixels(livecap_entry: LiveCapEntry) -> ndarray:
    """Turn the 2d keypoints into pixels(not rounded)."""
    kp_2d = livecap_entry.vibe.kp_2d
    cam = livecap_entry.vibe.theta[:3]
    h, w, _ = livecap_entry.frame.shape
    bbox = livecap_entry.bbox
    return kp_to_orig_image(h, w, cam, bbox, kp_2d)


def key_by_name_number(path: Path):
    return int(str(path.name).split('.')[0])


class LiveCapDataset:
    """Class for reading the LiveCap dataset."""
    props = ['frame', 'bbox', 'vibe', 'silhouette']

    def __init__(self, root: Union[str, Path], props_to_read: List[str] = None):
        """Creates a LiveCap dataset that can be accessed by indices.

        :param root: path to the root of the in the file system.
        :param props_to_read: if we are only interested to read some of the data
        """
        if isinstance(root, str):
            root = Path(root)
        # this will contain {'prop_name': [list of file paths],...}
        self.file_paths = {}
        for prop in self.props:
            dir_path = root / prop
            self.file_paths[prop] = sorted(list(dir_path.glob('*')), key=key_by_name_number)

        self.n = len(self.file_paths['frame'])
        # read the first image to see what the sizes are
        first_image = read_rgb_image_as_numpy(self.file_paths['frame'][0])
        self.image_height, self.image_width, _ = first_image.shape

    def __getitem__(self, i: int) -> LiveCapEntry:
        frame = self.file_paths['frame'][i]
        frame = read_rgb_image_as_numpy(frame)

        silhouette = self.file_paths['silhouette'][i]
        silhouette = read_image_as_numpy(silhouette)

        bbox = self.file_paths['bbox'][i]
        with bbox.open('rb') as f:
            bbox = pickle.load(f)

        vibe = self.file_paths['vibe'][i]
        with vibe.open('rb') as f:
            vibe = pickle.load(f)
        vibe = VibeEntry(**vibe)
        camera_params = vibe.theta[:3]
        vibe.kp_2d = kp_to_orig_image(self.image_height, self.image_width, camera_params,
                                            bbox, vibe.kp_2d)

        return LiveCapEntry(frame, bbox, vibe, silhouette)

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        if self._i < self.n:
            item = self[self._i]
            self._i = self._i + 1
            return item
        else:
            raise StopIteration
