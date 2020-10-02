from scipy.spatial.transform import Rotation
import numpy as np

from Scene import Camera


def create_transformation_matrix(translation: list, rotation: list, scale: float) -> np.array:
    result = np.eye(4)
    rotation = Rotation.from_euler('xyz', rotation, degrees=True).as_matrix()
    result[:3, :3] = rotation
    result[:3, 3] = translation
    # TODO not sure about the scale. should we just multiply it by the whole matrix?
    return scale*result


def create_view_matrix(camera: Camera) -> np.array:
    view_matrix = np.eye(4)
    rotation = Rotation.from_euler('xyz', [camera.pitch, camera.yaw, camera.roll], degrees=True).as_matrix()
    view_matrix[:3, :3] = rotation
    view_matrix[:3, 3] = [-x for x in camera.position]
    return view_matrix
