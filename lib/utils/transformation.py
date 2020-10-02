"""Module for dealing with the transformations."""

import numpy as np
from numpy import ndarray
from scipy.spatial.transform import Rotation

# TODO: use only this as match as possible
# TODO: test this


def homogenize(vector: ndarray, is_point: bool = True) -> ndarray:
    """Adds 1 or zeros to the last coordinate of the vectors. assumes that they are row stacked,
    or just one vector."""
    if not isinstance(vector, ndarray):
        raise ValueError

    n_dims = len(vector.shape)
    if n_dims not in {1, 2}:
        raise ValueError

    n_points = 1 if n_dims == 1 else vector.shape[0]
    append_value = 1 if is_point else 0

    if n_dims == 2:
        append_vector = np.full((n_points, 1), append_value, dtype=vector.dtype)
        return np.concatenate((vector, append_vector), axis=1)

    return np.concatenate((vector, [append_value]))


def dehomogenize(hom_point: ndarray, is_point: bool = True):
    """Divides by the last coordinate and removes it."""
    if not isinstance(hom_point, ndarray):
        raise ValueError

    n_dims = len(hom_point.shape)
    if n_dims not in {1, 2}:
        raise ValueError

    if not is_point:
        if n_dims == 2:
            return hom_point[:, :-1]
        return hom_point[:-1]

    if n_dims == 2:
        return hom_point[:, :-1] / hom_point[:, -1].reshape((-1, 1))
    return hom_point[:-1] / hom_point[-1]


class Transformation:
    """Class for representing transformations in 3d space."""
    def __init__(self, matrix: ndarray = np.identity(4)):
        if not isinstance(matrix, ndarray):
            raise ValueError
        if matrix.shape != (4, 4):
            raise ValueError
        self.matrix = matrix.copy()

    def copy(self):
        return Transformation(self.matrix.copy())

    @classmethod
    def from_rotation_translation(cls, rotation: Rotation, translation: ndarray):
        if not isinstance(rotation, Rotation):
            raise ValueError
        if not isinstance(translation, ndarray):
            raise ValueError
        if translation.shape != (3,):
            raise ValueError
        matrix = np.identity(4)
        matrix[:3, :3] = rotation.as_matrix()
        matrix[:3, 3] = translation
        return cls(matrix)

    @property
    def rotation(self):
        return Rotation.from_matrix(self.matrix[:3, :3])

    @rotation.setter
    def rotation(self, value: Rotation):
        if not isinstance(value, Rotation):
            raise ValueError
        self.matrix[:3, :3] = value.as_matrix()

    @property
    def translation(self):
        return self.matrix[:3, 3]

    @translation.setter
    def translation(self, value: ndarray):
        if not isinstance(value, ndarray):
            raise ValueError
        if value.shape != (3,):
            raise ValueError
        self.matrix[:3, 3] = value

    @property
    def angles(self):
        return self.rotation.as_euler('xyz')

    @angles.setter
    def angles(self, value: ndarray):
        if not isinstance(value, ndarray):
            raise ValueError
        if value.shape != (3,):
            raise ValueError
        self.rotation = Rotation.from_euler('xyz', value)

    @property
    def rotvec(self):
        return self.rotation.as_rotvec()

    @rotvec.setter
    def rotvec(self, value: ndarray):
        self.rotation = Rotation.from_rotvec(value)

    def as_matrix(self):
        return self.matrix

    def apply(self, vector: ndarray, is_point: bool = True) -> ndarray:
        """Apply a transformation on a vector

        :param vector: can be a single 3d or homo 4d or a raw stack of (n,3) or (n,4) vectors
        :param is_point: if it is a vector representing point or a direction
        """
        if not isinstance(vector, ndarray):
            raise ValueError
        n_dims = len(vector.shape)
        if n_dims not in {1, 2}:
            raise ValueError

        vector_dim = vector.shape[0] if n_dims == 1 else vector.shape[1]
        if vector_dim not in {3, 4}:
            raise ValueError
        is_hom = vector_dim == 4

        if not is_hom:
            vector = homogenize(vector, is_point)

        vector = np.matmul(vector, self.matrix.transpose())
        if is_hom:
            return vector

        return dehomogenize(vector, is_point)

    def compose(self, other):
        if not isinstance(other, Transformation):
            raise ValueError
        return Transformation(np.matmul(self.matrix, other.matrix))

    def inverse(self):
        """Returns the inverse transformation."""
        return Transformation(np.linalg.inv(self.matrix))

    def convert_basis(self, t_old_to_new, t_new_to_old=None, scale=1):
        if t_new_to_old is None:
            t_new_to_old = t_old_to_new.inverse()
        self.matrix = t_old_to_new.matrix @ self.matrix @ t_new_to_old.matrix
        self.translation = self.translation * scale
