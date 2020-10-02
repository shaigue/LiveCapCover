"""The data structures definitions."""
from typing import List, Tuple
import numpy as np
from numpy import ndarray

import config
from lib.LBS.Animation import KeyFrame, Animation
from lib.utils.joint import Joint
from lib.utils.transformation import Transformation


class FacePart:
    def __init__(self, line):
        self.name = line[0]
        self.parent = line[1]
        self.translation = np.array([float(line[3]), float(line[4]), float(line[5])])

    def __repr__(self):
        return f"<FacePart name={self.name}, parent={self.parent}, translation={self.translation}>"


class DOF:
    def __init__(self, line1, line2, line3):
        self.name = line1[0]
        self.has_limits = True if line2[0] == 'limits' else False
        if self.has_limits:
            self.min_val = float(line2[1])
            self.max_val = float(line2[2])
        self.joint = line3[0]

    def __str__(self):
        return f"<DOF name={self.name}, has_limits={self.has_limits}, joint={self.joint}>"

    def __repr__(self):
        return f"<DOF name={self.name}, has_limits={self.has_limits}, joint={self.joint}>"


class RawModelData:
    """ This is the resulting data structure that the reader will produce,
    will be used by the model's init function.

    Attributes:
        n_vertices(int) - number of vertices in the model
        n_faces(int) - number of faces(triangles) in the model
        n_bound_joints(int) - number of joints in the model that directly affect vertices.
        n_total_joints(int)[n_total_joints >= n_bound_joints] - number of joints that
        vertices(ndarray)[dtype=float, shape=(n_vertices, 3)]: 3d coordinates of the model's vertices
        faces(ndarray)[dtype=int, shape=(n_faces, 3)]: each row represents a triangle, where the entries in that row are
            the indices in the 'vertices' array of the vertices of that triangle.
        vertex_normals(ndarray)[dtype=float, shape=(n_vertices, 3)]: 3d coordinates of the vertices normals
        vertex_texture_coords(ndarray)[dtype=float, shape=(n_vertices, 2)]: 2d texture coordinates of the vertices
        weight_matrix(ndarray)[dtype=float, shape=(n_vertices, n_bounded_joints)]: sparse matrix where the entry
            [v, j] is the weight of vertex 'v' in the vertices array on joint 'j' in the bound_joint_list.
        joint_list(List[Joint])[len(joint_list) == n_total_joints]: list of the joints, they are
            ordered in the same order as the order in the weight matrix
        root_joint(Joint)[total number of joints in the tree == n_total_joints]: the root joint of the joint tree
        n_keyframes(int): number of keyframes in the animation
        face_indices(dict): a name and the index that represents that body part
    """

    def __init__(self, n_vertices: int, n_faces: int, n_bound_joints: int, n_total_joints: int,
                 vertices: ndarray, faces: ndarray, vertex_normals: ndarray, vertex_texture_coords: ndarray,
                 weight_matrix: ndarray, joint_list: List[Joint], root_joint: Joint, n_keyframes: int,
                 # for supporting the livecap model -
                 transformation_matrix: ndarray = config.t_blender_to_camera,
                 vertex_labels: ndarray = None,
                 texture: ndarray = None,
                 dofs: List[DOF] = None,
                 scale: float = 1,
                 ):
        self.n_vertices = n_vertices
        self.n_faces = n_faces
        self.n_bound_joints = n_bound_joints
        self.n_total_joints = n_total_joints
        self.vertices = vertices
        self.faces = faces
        self.vertex_normals = vertex_normals
        self.vertex_texture_coords = vertex_texture_coords
        self.weight_matrix = weight_matrix
        self.joint_list = joint_list
        self.root_joint = root_joint
        self.n_keyframes = n_keyframes
        self.face_indices = config.model_face_indices

        self.dofs = dofs
        self.vertex_labels = vertex_labels
        self.texture = texture

        self.scale = scale
        self._transform_model(transformation_matrix, scale)

    def _transform_model(self, transformation_matrix: ndarray, scale: float):
        t_old_to_new = Transformation(transformation_matrix)
        t_new_to_old = t_old_to_new.inverse()
        self.vertices = t_old_to_new.apply(self.vertices)
        self.vertices *= scale
        self.vertex_normals = t_old_to_new.apply(self.vertex_normals, is_point=False)

        for joint in self.joint_list:
            joint.convert_basis(t_old_to_new, t_new_to_old, scale)
            if joint.has_animation():
                poses = joint.animation.poses
                for i in range(len(poses)):
                    poses[i] = t_old_to_new.matrix @ poses[i] @ t_new_to_old.matrix
                joint.animation.poses = poses

    def reduce_weight_matrix(self) -> Tuple[ndarray, ndarray]:
        """Reduces the raw weight matrix to the top 3 joints that affect that vertex, and re-normalizes their weights.

            Returns a tuple with 2 entries:
                vertex_joints(ndarray)[shape=(n_vertices, 3), dtype=int]: the indices for each vertex of the joints that
                    affect it the most
                vertex_joint_weights(ndarray)[shape=(n_vertices, 3), dtype=float]: the re-normalized weights of the top
                    3 joints for each vertex
        """
        # sort the weight matrix by the joint axis, i.e. find the order of the joints with respect to it's weight for
        # each vertex
        weight_sort = self.weight_matrix.argsort(axis=1)
        # take the 3 with with the highest weights(last 3 for each vertex), and their weights
        vertex_joints = weight_sort[:, -3:]
        vertex_joints_weights = np.take_along_axis(self.weight_matrix, vertex_joints, axis=1)
        # re-normalize the weights to sum up to 1
        normalizing_factor = (1 / vertex_joints_weights.sum(axis=1)).reshape((-1, 1))
        vertex_joints_weights = vertex_joints_weights * normalizing_factor

        return vertex_joints, vertex_joints_weights

    def get_animation(self) -> Animation:
        # make sure that all of the timestamps are close in time
        timestamps = None
        for joint in self.root_joint.preorder_iterator():
            if joint.has_animation():
                if timestamps is None:
                    timestamps = joint.animation.timestamps
                elif np.any((timestamps != joint.animation.timestamps)):
                    raise RuntimeError("2 joints with different timestamps discovered.")

        # for each time stamp, collect all of the transformations
        keyframes = []
        for i in range(self.n_keyframes):
            timestamp = timestamps[i]
            t_list = []
            for j in range(self.n_total_joints):
                joint = self.root_joint.find_joint_by_index(j)
                if joint.has_animation():
                    t_list.append(Transformation(joint.animation.poses[i]))
                else:
                    t_list.append(joint.t_initial_joint_to_parent.copy())

            keyframes.append(KeyFrame(t_list, timestamp))

        return Animation(keyframes)

