"""Module for the camera parameters."""
from typing import List


import numpy as np
from numpy import ndarray

from lib.utils.transformation import homogenize, dehomogenize


def get_no_rotation_projection_matrix(fx: float, fy: float, tx: float, ty: float, tz: float, cx: float, cy: float):
    intrinsic = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    extrinsic = np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
    ])
    return np.matmul(intrinsic, extrinsic)


def project_points(points_3d: ndarray, projection_matrix: ndarray):
    points_3d_hom = homogenize(points_3d)
    points_2d_hom = np.matmul(points_3d_hom, projection_matrix.transpose())
    points_2d = dehomogenize(points_2d_hom)
    # swap the x, y axis
    return points_2d[..., [1, 0]]


class Camera:
    """A class to represent a camera."""

    def __init__(self, t_camera_to_world: ndarray, image_h: int, image_w: int, fx: float, fy: float,
                 u: float = None, v: float = None):
        if u is None:
            u = image_w / 2
        if v is None:
            v = image_h / 2

        self.t_camera_to_world = t_camera_to_world
        self.image_h = image_h
        self.image_w = image_w
        self.t_world_to_camera = np.linalg.inv(t_camera_to_world)
        self.fx = fx
        self.fy = fy
        self.u = u
        self.v = v
        camera_intrinsic = np.array([
            [fx, 0, u],
            [0, fy, v],
            [0, 0,  1]
        ])
        camera_extrinsic = self.t_world_to_camera[:3]
        self.projection_matrix = np.matmul(camera_intrinsic, camera_extrinsic)

    @property
    def image_shape(self):
        return self.image_w, self.image_h

    @staticmethod
    def vtk_to_matrix_camera_pose(vtk_camera_pose: List[tuple]):
        """returns the transformation from camera frame to world frame from vtk camera pose
        parameters:
         camera position    focus point         up vector
        [(x, y, z),         (fx, fy, fz),       (nx, ny, nz)]
        """
        camera_translation, focal_point, world_up_vector = vtk_camera_pose
        camera_translation = np.array(camera_translation)
        focal_point = np.array(focal_point)
        world_up_vector = np.array(world_up_vector)
        # calculate the camera unit vectors
        camera_z_axis = focal_point - camera_translation
        focal_length = np.linalg.norm(camera_z_axis)
        camera_z_axis = camera_z_axis / focal_length
        camera_y_axis = world_up_vector - world_up_vector.dot(camera_z_axis) * world_up_vector
        camera_y_axis = camera_y_axis / np.linalg.norm(camera_y_axis)
        # camera y axis points down in the image coordinates
        camera_y_axis = -camera_y_axis
        camera_x_axis = np.cross(camera_y_axis, camera_z_axis)

        t_camera_to_world = np.identity(4)
        t_camera_to_world[:3, 0] = camera_x_axis
        t_camera_to_world[:3, 1] = camera_y_axis
        t_camera_to_world[:3, 2] = camera_z_axis
        t_camera_to_world[:3, 3] = camera_translation

        return t_camera_to_world

    @staticmethod
    def matrix_to_vtk_camera_pose(matrix: ndarray):
        camera_translation = matrix[:3, 3]
        # add a z unit vector
        focal_point = camera_translation + matrix[:3, 2]
        # put the y unit vector, reversed
        world_up_vector = -matrix[:3, 1]
        return np.array([camera_translation, focal_point, world_up_vector])

    def to_vtk_camera_pose(self):
        return self.matrix_to_vtk_camera_pose(self.t_camera_to_world)

    def project(self, points: ndarray):
        return project_points(points, self.projection_matrix)
