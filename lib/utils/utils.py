"""Utilities functions."""
import numpy as np
from numpy import ndarray
import pickle
from pathlib import Path
from datetime import datetime
from PIL import Image


def array_elements_in_range(array: np.ndarray, lower: float, upper: float) -> bool:
    return (array >= lower).all() and (array <= upper).all()


def read_image_rgb(image_path):
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image


def read_rgb_image_as_numpy(image_path):
    return np.array(read_image_rgb(image_path))


def read_image_as_numpy(image_path):
    return np.array(Image.open(image_path))


def print_attr_types(obj):
    for attr in dir(obj):
        if attr.startswith('_'):
            continue
        print(f'type of {attr} is {type(getattr(obj, attr))}')
        if isinstance(getattr(obj, attr), ndarray):
            print(f'shape {getattr(obj, attr).shape}')
        if isinstance(getattr(obj, attr), (str, int)):
            print(f'value {getattr(obj, attr)}')
        if isinstance(getattr(obj, attr), (list, tuple)):
            print(f'len {len(getattr(obj, attr))}')


def save_object(object, dir_path: Path, file_name: str = None):
    dir_path.mkdir(parents=True, exist_ok=True)
    if file_name is None:
        file_name = 'animation_' + datetime.now().strftime("%y_%m_%d_%H_%M") + '.pkl'
    else:
        file_name += '.pkl'
    path = dir_path / file_name
    print('saving object: ' + str(path))

    with open(path, 'wb') as f:
        pickle.dump(object, f)
    return


def load_object(file_path: Path):
    with open(file_path, 'rb') as f:
        animation = pickle.load(f)
    return animation
