"""Read a collada file"""
from pathlib import Path
import re
from typing import Union, Dict, Tuple, List

from collada import Collada
from collada.controller import Skin
from collada.scene import Node, ExtraNode
from collada.triangleset import TriangleSet
from collada.animation import Animation
import numpy as np
from numpy import ndarray

from lib.utils.raw_model_data import RawModelData
from lib.utils.joint import Joint
from lib.utils.transformation import Transformation


def _check_inverse_bind_transform(joint_list: List[Joint], joint_sid_to_ibt_map: Dict[str, ndarray]):
    for joint in joint_list:
        if joint.name in joint_sid_to_ibt_map:
            inverse_bind_transform = joint_sid_to_ibt_map[joint.name]
            diff = np.abs(joint.t_model_to_joint.matrix - inverse_bind_transform).max()
            if diff > 1e-3:
                raise RuntimeError(f"got high diff in joint={joint}, diff={diff}")


def _find_skeleton_root_aux(node: Node, joint_sids: ndarray) -> Union[Node, None]:
    """Checks if a given node contains an sid with that name."""
    if isinstance(node, ExtraNode):
        return None

    attrib = node.xmlnode.attrib
    is_joint = attrib['type'] == 'JOINT'
    if is_joint and attrib['sid'] in joint_sids:
        return node

    for child in node.children:
        root = _find_skeleton_root_aux(child, joint_sids)
        if root is not None:
            if is_joint:
                return node
            return root

    return None


def _find_skeleton_root(node_list: List[Node], joint_sids: ndarray) -> Node:
    """Finds the first 'JOINT' node that has a child that is in 'joint_sids'."""
    for node in node_list:
        root = _find_skeleton_root_aux(node, joint_sids)
        if root is not None:
            return root

    raise RuntimeError('root node not find with the correct sid')


def _subtree_contains_sid(node: Node, joint_sids: ndarray) -> bool:
    if isinstance(node, ExtraNode):
        return False
    if 'sid' in node.xmlnode.attrib and node.xmlnode.attrib['sid'] in joint_sids:
        return True
    for child in node.children:
        if _subtree_contains_sid(child, joint_sids):
            return True
    return False


def _build_joint_tree(node: Node, joint_sids: ndarray) -> Joint:
    """Returns a root to a tree data structure of the Nodes converted to Joints.
    Trims all the nodes that are not parents to any of the important joints.
    """
    sid = node.xmlnode.attrib['sid']
    joint = Joint(sid, node.matrix)

    for child in node.children:
        if _subtree_contains_sid(child, joint_sids):
            child_joint = _build_joint_tree(child, joint_sids)
            joint.children.append(child_joint)

    return joint


def _check_animation(joint: Joint, n_keyframes: int):
    if joint.has_animation():
        if joint.animation.poses.shape[0] != joint.animation.timestamps.shape[0]:
            raise RuntimeError("found an animation with mismatch in its")
        if joint.animation.timestamps.shape[0] != n_keyframes:
            raise RuntimeError("found an animation with number of timestamps that does not match other animations.")
    for child in joint.children:
        _check_animation(child, n_keyframes)


def _check_all_vertices_in_faces(n_vertices: int, faces: ndarray):
    indices = np.arange(n_vertices)
    in_faces = np.in1d(indices, faces)
    if not np.all(in_faces):
        raise RuntimeError('not all the vertices are in the faces.')


def _get_vertex_texture_coords(n_vertices: int, faces: ndarray, texcoord_indexset: Tuple[ndarray],
                               texcoordset: Tuple[ndarray]) -> ndarray:
    vertex_indices = faces.reshape(-1)
    texture_indices = texcoord_indexset[0].reshape(-1)
    texture_coords = texcoordset[0]

    if len(vertex_indices) != len(texture_indices):
        raise RuntimeError("length of vertex indices do not match length of texture indices.")

    vertex_texture_coords = np.empty((n_vertices, 2), dtype=float)

    for i in range(len(vertex_indices)):
        vertex_i = vertex_indices[i]
        texture_i = texture_indices[i]
        vertex_texture_coords[vertex_i] = texture_coords[texture_i]

    return vertex_texture_coords


def _get_vertex_normals(n_vertices: int, faces: ndarray, normal_index: ndarray,
                        normal: ndarray) -> ndarray:
    vertex_indices = faces.reshape(-1)
    normal_indices = normal_index.reshape(-1)

    if len(vertex_indices) != len(normal_indices):
        raise RuntimeError("length of vertex indices do not match length of normal indices.")

    vertex_normals = np.empty((n_vertices, 3), dtype=float)

    for i in range(len(vertex_indices)):
        vertex_i = vertex_indices[i]
        normal_i = normal_indices[i]
        vertex_normals[vertex_i] = normal[normal_i]

    return vertex_normals


def _init_joint_indices(root_joint: Joint, joint_sids: ndarray) -> List[Joint]:
    """Assigns each of the joints in the tree with an index. first matches that of the ones in the list,
        and then creates new indices to match all of the other joints."""
    i = 0
    joint_list = []
    for joint_sid in joint_sids:
        joint = root_joint.find_joint_by_name(joint_sid)
        if joint is None:
            raise RuntimeError('joint did not match.')
        joint.index = i
        joint_list.append(joint)
        i += 1

    for joint in root_joint.preorder_iterator():
        if joint.index is None:
            joint.index = i
            joint_list.append(joint)
            i += 1

    return joint_list


def _get_joints_data(node_list: List[Node], controller: Skin) -> (Joint, List[Joint], int, int):
    """Reads the joint information from the collada object
    Returns 4 values:
        root_joint(Joint)
        joint_list(List[Joint])
        n_bound_joints(int)
        n_total_joints(int)
    """
    # find the joints that directly affect some vertices
    joint_sids = controller.weight_joints.data.flatten()
    n_bound_joints = len(joint_sids)
    # find the top node that contains any of the 'joint_sids', and that is of type 'JOINT':
    root = _find_skeleton_root(node_list, joint_sids)
    # build the joints hierarchy
    root_joint = _build_joint_tree(root, joint_sids)
    root_joint.set_root()
    # initialize the indices of the joints
    joint_list = _init_joint_indices(root_joint, joint_sids)
    # initialize the matrices
    root_joint.init_skeleton()
    # check that the calculated matrices match the original matrices
    _check_inverse_bind_transform(joint_list, controller.joint_matrices)
    n_total_joints = len(root_joint)
    return root_joint, joint_list, n_bound_joints, n_total_joints


def _get_weight_matrix(n_vertices: int, n_bound_joints: int, weights: ndarray, weight_indices: List[ndarray],
                       joint_indices: List[ndarray]) -> ndarray:
    """Create the weight matrix size (n_vertices, n_joints) where the entry [v, j] is the weight of vertex 'v' on
    joint index 'j'"""
    if not (len(joint_indices) == len(weight_indices) == n_vertices):
        raise RuntimeError(f'lengths of the joint indices and the weight indices should be the same as n_vertices.'
                           f' got {len(joint_indices)}, {len(weight_indices)}, {n_vertices}')

    weight_matrix = np.zeros((n_vertices, n_bound_joints), dtype=float)
    for v in range(n_vertices):
        if joint_indices[v].shape != weight_indices[v].shape:
            raise RuntimeError(f"{joint_indices[v].shape} should be the same as {weight_indices[v].shape}.")
        weight_matrix[v, joint_indices[v]] = weights[weight_indices[v]]

    if not np.allclose(weight_matrix.sum(axis=1), 1):
        raise RuntimeError("weights are not normalized to 1.")

    return weight_matrix


def _read_animations(animations: List[Animation], root_joint: Joint) -> int:
    # read animation key frames:
    n_keyframes = 0
    if len(animations) > 1:
        raise RuntimeError(f'expected to get up to a single animation, got {len(animations)}')

    animation_node: Animation = animations[0]
    parser = re.compile(r'.*Action_(\d{3}_)?(.*)_pose_matrix-(input|output|interpolation)')

    for animation_id, source in animation_node.sourceById.items():
        match_obj = parser.match(animation_id)
        if match_obj:
            _, sid, source_type = match_obj.groups()
            joint = root_joint.find_joint_by_name(sid)
            if joint:
                # timestamp
                if source_type == 'input':
                    joint.animation.timestamps = source.data.reshape(-1)
                    n_keyframes = joint.animation.timestamps.shape[0]
                # pose
                elif source_type == 'output':
                    joint.animation.poses = source.data.reshape((-1, 4, 4))
        else:
            raise RuntimeError(f'got an animation that did not match the regexp {animation_id}')

    _check_animation(root_joint, n_keyframes)
    return n_keyframes


def read_collada_file(model_path: Union[Path, str]) -> RawModelData:
    """Reads the model and returns it's data in the desirable format."""
    if isinstance(model_path, Path):
        model_path = str(model_path)
    collada_obj = Collada(model_path)

    # assumes that there is a single controller
    if len(collada_obj.controllers) != 1:
        raise RuntimeError(f'there should be exactly 1 controller, got {len(collada_obj.controllers)}')
    controller: Skin = collada_obj.controllers[0]
    triangles: TriangleSet = controller.geometry.primitives[0]

    # get vertices
    vertices = triangles.vertex
    n_vertices = vertices.shape[0]
    # apply the bind_shape_matrix to all of the vertices
    t_bind_shape = Transformation(controller.bind_shape_matrix)
    vertices = t_bind_shape.apply(vertices)
    # get faces
    faces = triangles.vertex_index
    n_faces = triangles.ntriangles
    _check_all_vertices_in_faces(n_vertices, faces)
    # get texture
    vertex_texture_coords = _get_vertex_texture_coords(n_vertices, faces, triangles.texcoord_indexset,
                                                       triangles.texcoordset)
    # get normals
    vertex_normals = _get_vertex_normals(n_vertices, faces, triangles.normal_index, triangles.normal)

    # get joint data
    root_joint, joint_list, n_bound_joints, n_total_joints = _get_joints_data(collada_obj.scene.nodes, controller)

    # get weight matrix
    weight_matrix = _get_weight_matrix(n_vertices, n_bound_joints, controller.weights.data.flatten(),
                                       controller.weight_index, controller.joint_index)
    # read animation data
    animations = collada_obj.animations
    n_keyframes = 0
    if len(animations) > 0:
        n_keyframes = _read_animations(animations, root_joint)

    return RawModelData(
        n_vertices=n_vertices, n_faces=n_faces, n_bound_joints=n_bound_joints, n_total_joints=n_total_joints,
        vertices=vertices, faces=faces, vertex_normals=vertex_normals, vertex_texture_coords=vertex_texture_coords,
        weight_matrix=weight_matrix, joint_list=joint_list, root_joint=root_joint,
        n_keyframes=n_keyframes,
    )
