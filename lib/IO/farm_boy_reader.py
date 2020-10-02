from typing import List
from pathlib import Path
import numpy as np
import collada

from lib.utils.raw_model_data import Joint
from lib.LBS.Animation import KeyFrame, Transformation, Animation
from lib.LBS.Model import Model


def read(farm_boy_model_path: Path) -> Model:
    '''
    Read a collada file.

    When reading a collada file we need to extract data about two main parts of the model - the mesh and the skeleton. We can also
    extract animation data.

    The mesh data includes the vertices, texture_coords and normals. It also includes the indices.
    The skeleton data includes information about the skeleton of the model - the joints, their hierarchy, transformation matrices (local
    bind transform, inverse bind transform); also, for each vertex we have the joints that affect it and the respective weights (used for LBS).
    The animation data mainly includes the key frames.
    '''

    ext = farm_boy_model_path.suffix.lower()
    if ext != '.dae':
        print("Please use only .dae files. Exiting...")
        raise SystemExit

    model_data = collada.Collada(farm_boy_model_path.as_posix())
    data = _read_model_data(model_data)

    model = Model(
        faces=data["faces"], vertices=data["vertices"], vertex_texture_coords=data["vertex_texture_coords"],
        vertex_normals=data["vertex_normals"], root_joint=data["root_joint"], vertex_joints=data["vertex_joints"],
        vertex_joints_weights=data["vertex_joints_weights"], animation=Animation(data["key_frames"])
    )
    return model


def _read_model_data(model_data):
    '''
    Currently, we don't read the texture coordinates here for simplicity. The reason is that texture coordinates have their own
    indices which usually don't match to the vertex indices, but opengl's index buffer requires them to match. In short this means that
    for each unique combination of a vertex and a texture coordinate we need to create an index for it, which requires some work.
    Basically the purpose of the whole indices idea and the index buffer is to save up memory, so maybe we can just duplicate the
    vertices and have one index for each vertex... but it might slow down everything.
    '''
    data = {}

    triset = model_data.geometries[0].primitives[0]

    # mesh
    data["faces"] = np.arange(len(triset.vertex_index)).reshape(-1, 3)
    data["vertices"] = triset.vertex[triset.vertex_index]
    data["vertex_texture_coords"] = triset.texcoordset[0][triset.texcoord_indexset[0]]
    data["vertex_normals"] = np.zeros(data["vertices"].shape)

    # skeleton
    data["vertex_joints"], data["vertex_joints_weights"] = _read_vertex_joints_weights(model_data)
    data["root_joint"]: Joint = _read_joints(model_data)

    # animation
    data["key_frames"] = _read_animation(model_data)

    return data


def _read_vertex_joints_weights(model_data):
    weights = model_data.controllers[0].weights
    index = model_data.controllers[0].index

    index = [np.vstack((a[:, 0], weights[a[:, 1]].flatten())).T for a in index]         # replace weight indices with weights
    index = [a[a[:, 1].argsort()][::-1] for a in index]         # sort by weights, descending order
    index = [np.pad(a, ((0, 3), (0, 0))) for a in index]        # append 3 zero rows (can append 2, but just in case)
    index = [a[:3, :] for a in index]                           # keep only 3 joints
    index = [np.vstack((a[:, 0],  a[:, 1]/np.sum(a, axis=0)[1])).T for a in index]      # normalize sum to 1.0
    index = np.array(index)

    vertex_joints = index[:, :, 0]
    weights = index[:, :, 1]

    triset = model_data.geometries[0].primitives[0]
    vertex_joints = vertex_joints[triset.vertex_index]
    weights = weights[triset.vertex_index]
    return vertex_joints, weights


def _read_joints(model_data) -> Joint:
    joints = []
    joints_inv_bind_matrix = model_data.controllers[0].joint_matrices
    inv_bind_matrices = []
    # TODO not sure if the dictionary is always ordered
    # we rely on this order, because the vertex weights are indexed by this order of the joints
    for index, name in enumerate(joints_inv_bind_matrix):
        inv_bind_matrices.append(joints_inv_bind_matrix[name])
        joint = Joint(name, np.eye(4), index)
        joints.append(joint)

    # TODO currently we can only read the farm boy model... it seems there is no convention for which node has the hierarchy information
    root_node = model_data.scenes[0].nodes[1].children[0]
    root_joint: Joint = _build_joint_tree(root_node, joints)
    root_joint.set_root()
    assert len(root_joint) == len(joints)
    root_joint.init_skeleton()
    _test_t_model_to_joint(root_joint, inv_bind_matrices)
    return root_joint


def _test_t_model_to_joint(joint, inv_bind_matrices):
    '''
    '''
    if not np.allclose(joint.t_model_to_joint, inv_bind_matrices[joint.index], atol=0.0001):
        raise RuntimeError()
    for child in joint.children:
        _test_t_model_to_joint(child, inv_bind_matrices)


def _read_animation(model_data):
    animations = model_data.animations

    timestamps = animations[0].sourceById['Armature_Torso_pose_matrix-input'][:, 0]

    all_transforms = np.zeros((len(animations), len(timestamps), 16))
    for i, joint_animation in enumerate(animations):
        transforms_data = [value for key, value in joint_animation.sourceById.items() if 'output' in key][0]
        joint_transforms = transforms_data[:, 0].reshape(-1, 16)
        assert len(joint_transforms) == len(timestamps)
        all_transforms[i] = joint_transforms

    key_frames = []
    for i, timestamp in enumerate(timestamps):
        # get transforms of all joints in timestamp[i]
        current_transforms = all_transforms[:, i]

        transforms: List[Transformation] = []
        for i in range(len(current_transforms)):
            # get transform of all joints in current key frame
            joint_current_transform = current_transforms[i].reshape(4, 4)
            joint_current_transform: Transformation = Transformation.from_matrix(joint_current_transform)
            transforms.append(joint_current_transform)

        kf = KeyFrame(transforms, timestamp)
        key_frames.append(kf)

    return key_frames


def _build_joint_tree(root_node, joints) -> Joint:
    root_joint: Joint = _find_joint_by_name(joints, root_node.id)
    root_joint.t_joint_to_parent = root_node.matrix
    for child in root_node.children:
        if hasattr(child, 'id'):
            j = _build_joint_tree(child, joints)
            root_joint.add_child(j)
    return root_joint


def _find_joint_by_name(joints, name: str) -> Joint:
    return [j for j in joints if j.name == name][0]
