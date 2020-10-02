import numpy as np
from numpy import ndarray
import logging

from lib.data_utils.livecap_dataset_adapter import AdapterEntry
from lib.utils.model import Model
from lib.utils.camera import Camera
from lib.utils.renderer import Renderer, DebugRenderer
from lib.utils.anatomic_angle_limits import load_joint_angle_limits
from lib.image_processing.image_distance_transform import image_distance_transform

logging.basicConfig(filename='energy.log', filemode='w', level=logging.DEBUG,
                    format="$(asctime)s-%(module)s-%(funcName)s-%(message)s")
num_eval = 0


class Energy:
    def __init__(self, model: Model, camera: Camera, joint_indices: ndarray, renderer: Renderer,
                 c_2d: float = 0, c_3d: float = 0, c_anatomic: float = 0, c_silhouette: float = 0,
                 c_temporal: float = 0, return_scalar: bool = False, debug: bool = False):
        logging.debug('')
        self.model = model
        self.camera = camera
        self.renderer = renderer
        self.joint_indices = joint_indices
        self.c_2d = c_2d
        self.c_3d = c_3d
        self.c_anatomic = c_anatomic
        self.c_silhouette = c_silhouette
        self.c_temporal = c_temporal
        self.prev_p3d = None
        self.min_joint_angles, self.max_joint_angles = load_joint_angle_limits(model)
        self.return_scalar = return_scalar
        self.debug = debug
        self.debug_renderer = DebugRenderer(camera.image_w, camera.image_h, model, joint_indices)

    def frame_end(self):
        """saves the current joint position."""
        self.prev_p3d = self.model.get_p3d()

    def show_debug(self):
        return self.debug

    def energy_2d(self, p3d: ndarray, kp_2d: ndarray) -> ndarray:
        # project the points into the image coordinates
        p2d = self.camera.project(p3d)
        if self.show_debug():
            self.debug_renderer.debug_2d(p2d, kp_2d)
        return p2d - kp_2d

    def energy_3d(self, p3d: ndarray, kp_3d: ndarray, kp_3d_translation: ndarray) -> ndarray:
        # return their difference with with the given translated positions
        if self.show_debug():
            self.debug_renderer.debug_3d(p3d, kp_3d, kp_3d_translation)
        return p3d - (kp_3d + kp_3d_translation)

    def energy_silhouette(self, silhouette: ndarray) -> ndarray:
        # TODO: is there a better way to calculate the silhouette loss
        # TODO: to calculate the b_i's from the article we need to find ut what vertices are in the
        #  inside and what are outside, the b_i can only be negative for thoes on the inside
        model_silhouette = self.renderer.draw_model_silhouette()
        # calculate the IDT
        idt = image_distance_transform(silhouette)
        # count the number of silhouette pixels
        n_silhouette_pixels = model_silhouette.sum()
        # if it is zero return the maximum possible value in idt
        if n_silhouette_pixels == 0:
            if self.show_debug():
                self.debug_renderer.bad_silhouette()
            return np.array([idt.max()])
        # sum all of the IDT in the silhouette, and average them
        average_idt = (idt * model_silhouette).sum() / n_silhouette_pixels
        if self.show_debug():
            self.debug_renderer.debug_silhouette(model_silhouette, silhouette, n_silhouette_pixels, idt, average_idt)

        return np.array([average_idt])

    def energy_temporal(self, p3d: ndarray) -> ndarray:
        if self.prev_p3d is None:
            return np.zeros_like(p3d)
        return p3d - self.prev_p3d[self.joint_indices]

    def energy_anatomic(self, pose_vector: ndarray) -> ndarray:
        # drop the translation
        angles = pose_vector[3:]
        result = np.where(angles > self.max_joint_angles, angles - self.max_joint_angles, 0)
        result = np.where(angles < self.min_joint_angles, self.min_joint_angles - angles, result)
        return result

    def set_debug(self, debug: bool):
        self.debug = debug

    def energy_pose(self, pose_vector: ndarray, entry: AdapterEntry) -> ndarray:
        # apply the new pose, and get the joint's positions
        self.model.apply_pose_vector(pose_vector)
        p3d = self.model.get_p3d()
        p3d = p3d[self.joint_indices]
        # collect all the energy terms
        energies = (
            self.c_2d * self.energy_2d(p3d, entry.kp_2d),
            self.c_3d * self.energy_3d(p3d, entry.kp_3d, entry.kp_3d_translation),
            # self.c_silhouette * self.energy_silhouette(entry.silhouette),
            self.c_anatomic * self.energy_anatomic(pose_vector),
            self.c_temporal * self.energy_temporal(p3d),
        )
        # flatten and concatenate it into a single vector
        result = np.concatenate(energies, axis=None)
        if self.return_scalar:
            return (result**2).sum()
        return result
