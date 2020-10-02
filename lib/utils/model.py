"""Module that encapsulates the 3d model."""

from pathlib import Path
from typing import Union, List, Optional

import numpy as np
from numpy import ndarray

from lib.IO.read_collada import read_collada_file
from lib.utils.raw_model_data import RawModelData, DOF, FacePart
from lib.utils.joint import Joint
from lib.utils.transformation import Transformation, homogenize, dehomogenize


def linear_blend_skinning(vertices: ndarray, normals: ndarray, joints_list: List[Joint], weight_matrix: ndarray):
    """
    :param vertices: (n_vertices, 3)
    :param joints_list: (n_joints)
    :param weight_matrix: (n_vertices, n_joints)
    """
    # validate input:
    if vertices.shape[1] != 3:
        raise ValueError(f'vertices should be (n, 3) got {vertices.shape}')
    n_joints = weight_matrix.shape[1]
    n_vertices = vertices.shape[0]
    if n_joints > len(joints_list):
        raise ValueError(f'n_bounded_joints should be smaller then the size of the joints list, '
                         f'got {n_joints}, joint_list={len(joints_list)}')
    if weight_matrix.shape != (n_vertices, n_joints):
        raise ValueError(f'weight matrix should be of shape {(n_vertices, n_joints)}, got {weight_matrix.shape}')
    # take only the effective joints
    joints_list = joints_list[:n_joints]
    # collect all the flattened joint matrices in a single matrix with shape (n_joints, 16)
    joint_matrices = np.stack([joint.t_joint.matrix.reshape(-1) for joint in joints_list])
    # calculate for each vertex it's weighted mean of the joint matrices according to the weights in the 'weight_matrix'
    # we will get (n_vertices, 16) matrix
    vertex_matrices = np.matmul(weight_matrix, joint_matrices)
    # reshape the matrices to be (n_vertices, 4, 4)
    vertex_matrices = vertex_matrices.reshape((-1, 4, 4))
    # homogenize the vertices (n_vertices, 4)
    vertices = homogenize(vertices)
    normals = homogenize(normals, is_point=False)
    # 8. calculate U'[i] = T[i] @ V[i]'
    vertices = np.einsum('...jk, ...k', vertex_matrices, vertices)
    normals = np.einsum('...jk, ...k', vertex_matrices, normals)
    # dehomogenize the vertices and return them
    return dehomogenize(vertices), dehomogenize(normals, is_point=False)


class LivecapPose:
    """Class that returns a result for the corresponding pose for a given frame.

    Attributes:
        root_translation(ndarray)[shape=(3, ), dtype=float]: the translation of the root joint of the model
        angles(ndarray)[shape=(n_joints, 3), dtype=float]: the angles for each joint in x, y, z axis accordingly.

    Notes:
        vector form - root_translation concatenated with the angles like this:
            3                   3               3                   3
            root_translation    joint_1 angles  joint_2 angles ...  joint_n angles
    """

    def __init__(self, root_translation: ndarray, angles: ndarray):
        if not isinstance(root_translation, ndarray):
            raise ValueError
        if root_translation.shape != (3,):
            raise ValueError
        if not isinstance(angles, ndarray):
            raise ValueError
        if len(angles.shape) != 2:
            raise ValueError
        if angles.shape[1] != 3:
            raise ValueError
        self.root_translation = root_translation
        self.angles = angles

    def to_vector(self) -> ndarray:
        """Return in a vector form."""
        return np.concatenate((self.root_translation, self.angles), axis=None)

    @classmethod
    def from_vector(cls, vector: ndarray):
        """initialize from vector form.
        """
        root_translation = vector[:3]
        angles = vector[3:].reshape((-1, 3))
        return cls(root_translation, angles)

    @classmethod
    def from_transformation_list(cls, transformation_list: List[Transformation], root_index: int = 0):
        root_translation = transformation_list[root_index].translation
        angles = np.empty((len(transformation_list), 3))
        for i, transformation in enumerate(transformation_list):
            angles[i] = transformation.angles
        return cls(root_translation, angles)


class Model:
    def __init__(self, root_joint: Joint, joint_list: List[Joint], initial_vertices: ndarray, faces: ndarray,
                 normals: ndarray, weight_matrix: ndarray, n_bound_joints: int, face_indices: dict = None, ):
        self.root_joint = root_joint
        self.joint_list = joint_list
        self.initial_vertices = initial_vertices
        self.vertices = initial_vertices.copy()
        self.faces = faces
        self.initial_normals = normals
        self.normals = normals.copy()
        self.weight_matrix = weight_matrix
        self.n_bound_joints = n_bound_joints
        self.n_joints = len(joint_list)
        self.face_indices = face_indices
        self.face_parts_list = list(face_indices.keys()) if face_indices is not None else None
        self.skeleton_connectivity = self._init_skeleton_connectivity()

    @classmethod
    def from_raw_model_data(cls, raw_model_data: RawModelData):
        return cls(
            root_joint=raw_model_data.root_joint,
            joint_list=raw_model_data.joint_list,
            initial_vertices=raw_model_data.vertices,
            faces=raw_model_data.faces,
            normals=raw_model_data.vertex_normals,
            weight_matrix=raw_model_data.weight_matrix,
            n_bound_joints=raw_model_data.n_bound_joints,
            face_indices=raw_model_data.face_indices,
        )

    @classmethod
    def from_file(cls, file_path: Union[Path, str]):
        raw_model_data = read_collada_file(file_path)
        return cls.from_raw_model_data(raw_model_data)

    def update_vertices(self):
        self.root_joint.calc_t_joint()
        self.vertices, self.normals = linear_blend_skinning(
            vertices=self.initial_vertices,
            normals=self.initial_normals,
            joints_list=self.joint_list,
            weight_matrix=self.weight_matrix,
        )

    def get_initial_pose(self) -> LivecapPose:
        """Returns the initial pose of the model."""
        root_translation = self.root_joint.initial_translation
        angles = np.empty((self.n_joints, 3))

        for i, joint in enumerate(self.joint_list):
            angles[i] = joint.initial_angles

        return LivecapPose(root_translation, angles)

    def apply_livecap_pose(self, livecap_pose: LivecapPose):
        self.root_joint.set_joint_to_parent_translation(livecap_pose.root_translation)

        for i, joint in enumerate(self.joint_list):
            joint.set_joint_to_parent_angles(livecap_pose.angles[i])

        self.update_vertices()

    def apply_pose_vector(self, pose_vector: ndarray):
        self.apply_livecap_pose(LivecapPose.from_vector(pose_vector))

    def apply_transformation_list(self, transformation_list: List[Transformation]):
        self.apply_livecap_pose(LivecapPose.from_transformation_list(transformation_list, self.root_joint.index))

    def get_p3d(self):
        """Returns the 3d joints positions concatenated after with the face positions."""
        joints = self.get_joints_positions()
        face = self.get_face_vertices()
        return np.concatenate((joints, face), axis=0)

    def get_joints_positions(self) -> ndarray:
        joints_positions = np.empty((self.n_joints, 3))
        for i, joint in enumerate(self.joint_list):
            joints_positions[i] = joint.world_coordinates
        return joints_positions

    def get_face_vertices(self) -> ndarray:
        face_vertices = np.empty((len(self.face_parts_list), 3))
        for i, face_part_name in enumerate(self.face_parts_list):
            index = self.face_indices[face_part_name]
            face_vertices[i] = self.vertices[index]
        return face_vertices

    def _init_skeleton_connectivity(self) -> ndarray:
        edges = []
        for joint in self.joint_list:
            for child in joint.children:
                edges.append([joint.index, child.index])
        return np.array(edges)

    def get_root_joint_index(self) -> int:
        return self.root_joint.index

    def get_joint_index_from_name(self, name: str) -> int:
        for joint in self.joint_list:
            if joint.name == name:
                return joint.index


class LivecapModel(Model):
    """A class that takes the special structure of the livecap model, especially the single rotation axis."""

    def __init__(self, root_joint: Joint, joint_list: List[Joint], initial_vertices: ndarray, faces: ndarray,
                 normals: ndarray, weight_matrix: ndarray, n_bound_joints: int,
                 # this is for supporting the structure of the new model
                 dofs: List[DOF],
                 texture: ndarray,
                 vertex_texture_coords: ndarray,
                 scale: float,
                 ):
        super().__init__(
            root_joint=root_joint,
            joint_list=joint_list,
            initial_vertices=initial_vertices,
            faces=faces,
            normals=normals,
            weight_matrix=weight_matrix,
            n_bound_joints=n_bound_joints
        )
        self.n_dof = len(dofs)
        self.dof_names = np.empty(self.n_dof, dtype=str)
        self.dof_joints = np.empty(self.n_dof, dtype=object)
        self.dof_limits_min = np.empty(self.n_dof, dtype=float)
        self.dof_limits_max = np.empty(self.n_dof, dtype=float)
        for i, dof in enumerate(dofs):
            self.dof_names[i] = dof.name
            self.dof_joints[i] = dof.joint
            if dof.has_limits:
                self.dof_limits_min[i] = dof.min_val
                self.dof_limits_max[i] = dof.max_val
            else:
                self.dof_limits_min[i] = -np.inf
                self.dof_limits_max[i] = np.inf

        self.texture = texture
        self.vertex_texture_coords = vertex_texture_coords
        self.scale = scale

    @classmethod
    def from_raw_model_data(cls, raw: RawModelData):
        return cls(
            root_joint=raw.root_joint,
            joint_list=raw.joint_list,
            initial_vertices=raw.vertices,
            faces=raw.faces,
            normals=raw.vertex_normals,
            weight_matrix=raw.weight_matrix,
            n_bound_joints=raw.n_bound_joints,
            dofs=raw.dofs,
            texture=raw.texture,
            vertex_texture_coords=raw.vertex_texture_coords,
            scale=raw.scale,
        )

    def get_limits(self):
        return self.dof_limits_min, self.dof_limits_max

    def apply_pose_vector(self, pose_vector: ndarray):
        if pose_vector.shape != (self.n_dof, ):
            raise ValueError

        for i in range(self.n_dof):
            joint: Joint = self.dof_joints[i]
            value = pose_vector[i]
            joint.apply_along_axis(value)

        super(LivecapModel, self).update_vertices()

    def get_p3d(self):
        """Returns the 3d joints positions concatenated after with the face positions."""
        return self.get_joints_positions()

    # ====================== NOT IMPLEMENTED ON PURPOSE ===============================
    @classmethod
    def from_file(cls, file_path: Union[Path, str]):
        raise NotImplemented

    def get_face_vertices(self) -> ndarray:
        raise NotImplemented

    def apply_livecap_pose(self, livecap_pose: LivecapPose):
        raise NotImplemented


