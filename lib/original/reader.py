"""Module to read the original livecap data and model."""
from pathlib import Path
from typing import List
import re
from itertools import chain

import numpy as np
from numpy import ndarray
from PIL import Image

from lib.utils.raw_model_data import RawModelData, DOF, FacePart
from lib.utils.joint import Joint, MovementType
from lib.utils.transformation import Transformation
from lib.data_utils.livecap_dataset_adapter import get_vibe_model_dimensions
from lib.data_utils.livecap_dataset import LiveCapDataset

# =================== Utility Functions ===========================


def split_line_clean(line: str):
    # remove the last character \n
    if line and line[-1] == '\n':
        line = line[:-1]
    # split with ' '
    line = line.split(' ')
    # remove empty strings and spaces
    line = list(filter(lambda x: x != '' and not x.isspace(), line))
    return line


def convert_string_list_to_float(str_list: List[str]):
    return list(map(float, str_list))


# ========================= Motion =============================================


def read_motion(path: Path) -> ndarray:
    """For each frame, finds the calculated motion."""
    with path.open('r') as f:
        # skip the header
        f.readline()
        # read all the file
        text = f.read()
    # split it into rows
    rows = text.splitlines(keepends=False)
    # split every row into values separated with ' '
    rows = map(lambda row: row.split(' '), rows)
    # remove '' from each row
    rows = map(lambda row: filter(lambda x: x != '', row), rows)
    # convert the strings into floats
    rows = map(lambda row: map(float, row), rows)
    # convert everything
    rows = list(map(list, rows))
    # remove the first index column
    return np.array(rows)[:, 1:]


# ======================= Calibration ==========================================


class Calibration:
    def __init__(self, width: int = None, height: int = None, intrinsic: ndarray = None, extrinsic: ndarray = None):
        self.width = width
        self.height = height
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic


def read_calibration(path: Path) -> List[Calibration]:
    """Returns a list of the calibrations read from file"""
    with path.open('r') as f:
        # remove the first line
        f.readline()
        calibration = None
        calibration_list = []
        for line in f.readlines():
            # split the line with ' ', and remove any empty or space
            line = split_line_clean(line)
            # if the first item is 'name', then create a new item
            if line[0] == 'name':
                calibration = Calibration()
            if line[0] == 'size':
                calibration.width = int(line[1])
                calibration.height = int(line[2])
            if line[0] == 'intrinsic':
                calibration.intrinsic = np.array(convert_string_list_to_float(line[1:]))
                calibration.intrinsic = calibration.intrinsic.reshape((4, 4))
            if line[0] == 'extrinsic':
                calibration.extrinsic = np.array(convert_string_list_to_float(line[1:]))
                calibration.extrinsic = calibration.extrinsic.reshape((4, 4))
                calibration_list.append(calibration)

    return calibration_list


# =================================== Skeleton ============================================
class JointData:
    def __init__(self, name, movement_type, parent, tx, ty, tz, nx, ny, nz, scale_factor):
        self.name = name
        self.movement_type = None
        if movement_type == 'prismatic':
            self.movement_type = MovementType.translation
        if movement_type == 'revolute':
            self.movement_type = MovementType.rotation

        if parent == 'none':
            self.parent = None
        else:
            self.parent = parent
        self.translation = np.array([float(tx), float(ty), float(tz)])
        self.rotation_axis = np.array([float(nx), float(ny), float(nz)])

    def __str__(self):
        return f"<JointData name={self.name}>"

    def __repr__(self):
        return f"<JointData name={self.name}>"


def split_parts(text: str) -> dict:
    """Reads and separates the parts of the file."""
    regexp = re.compile(r'\n(.*): ([0-9]*)')
    result = {}
    last_key = None
    for match in regexp.finditer(text):
        key, count = match.groups()
        start = match.end()
        result[key] = {'count': count, 'start': start, 'end': None}
        if last_key is not None:
            result[last_key]['end'] = match.start()
        last_key = key
    if last_key is not None:
        result[last_key]['end'] = len(text)

    for key in result.keys():
        result[key]['text'] = text[result[key]['start']:result[key]['end']]

    return result


def split_lines_and_remove_spaces(text: str):
    text = text.splitlines(keepends=False)
    lines = []
    for line in text:
        line = split_line_clean(line)
        if len(line) > 0:
            lines.append(line)
    return lines


def read_skeleton(path: Path):
    text = path.read_text()
    parts = split_parts(text)
    joints_lines = split_lines_and_remove_spaces(parts['joints']['text'])

    joints = []
    for joint_line in joints_lines:
        joints.append(JointData(*joint_line))

    dofs_lines = split_lines_and_remove_spaces(parts['dofs']['text'])
    dofs = []
    step = 3
    # print(dofs_lines)
    for i in range(0, len(dofs_lines), step):
        dofs.append(DOF(*dofs_lines[i:i+step]))

    markers_lines = split_lines_and_remove_spaces(parts['markers']['text'])
    face_parts_names = ['lefteye', 'righteye', 'nose', 'chin']
    face_parts = []
    for line in markers_lines:
        if line[0] in face_parts_names:
            face_parts.append(FacePart(line))

    return joints, dofs, face_parts


def read_skinning_data(skin_path: Path, skeleton_path: Path) -> dict:
    # Notes:
    # make sure that the joints that are bounded to vertices are first, and the indices are correct
    joints, dofs, face_parts = read_skeleton(skeleton_path)
    bone_names, weight_matrix = read_skin(skin_path)

    # build the joint tree
    # create the joints, and a mapping from name to joint
    name_to_joint = {}
    for joint_data in joints:
        t_joint_to_parent = Transformation()
        t_joint_to_parent.translation = joint_data.translation
        joint = Joint(name=joint_data.name, t_joint_to_parent=t_joint_to_parent.matrix,
                      axis=joint_data.rotation_axis, movement_type=joint_data.movement_type)
        name_to_joint[joint.name] = joint
    # add face parts as joints
    for face_part in face_parts:
        t_joint_to_parent = Transformation()
        t_joint_to_parent.translation = face_part.translation
        joint = Joint(name=face_part.name, t_joint_to_parent=t_joint_to_parent.matrix)
        name_to_joint[joint.name] = joint

    # make the dof and face_part point at the correct joint
    for dof in dofs:
        dof.joint = name_to_joint[dof.joint]
    # create the the joint tree, and find the root of the tree
    root = None
    for joint_data in chain(joints, face_parts):
        joint = name_to_joint[joint_data.name]
        if joint_data.parent is None:
            joint.set_root()
            root = joint
        else:
            parent = name_to_joint[joint_data.parent]
            parent.children.append(joint)

    assert root is not None
    root.init_skeleton()

    # match the joints in the bones list to the last joint in the hierarchy that matches that joint
    bones_to_joints = {}
    name_extractor = re.compile(r'(.*)_[a-z]*')

    def find_deepest_match(joint: Joint):
        name = name_extractor.match(joint.name)
        if name:
            name = name.group(1)
            bones_to_joints[name] = joint
        for child in joint.children:
            find_deepest_match(child)

    find_deepest_match(root)

    assert set(bone_names).issubset(bones_to_joints.keys())

    # order the list with the first joints
    joint_list = []
    for i, bone_name in enumerate(bone_names):
        joint = bones_to_joints[bone_name]
        joint.index = i
        joint_list.append(joint)

    i = len(joint_list)
    n_bound_joints = i
    # then index all of the other joints
    for joint in name_to_joint.values():
        if joint.index is not None:
            continue
        joint.index = i
        i += 1
        joint_list.append(joint)

    n_joints = len(joint_list)

    return {
        'n_bound_joints': n_bound_joints,
        'n_total_joints': n_joints,
        'weight_matrix': weight_matrix,
        'joint_list': joint_list,
        'dofs': dofs,
        'root_joint': root,
    }


# =================================== Skin ======================================


def read_skin(path: Path) -> (list, ndarray):
    """Returns a list with the bones names, and the weight matrix"""
    bone_names_line = 2
    vertex_weights_line = 4

    text = path.read_text()
    lines = text.splitlines(keepends=False)
    bone_names = split_line_clean(lines[bone_names_line])
    n_bones = len(bone_names)
    # for each line after 'vertex_weights_line' there is a vertex index, then pairs of joint_index, weight
    vertices_weights = {}
    for line in lines[vertex_weights_line:]:
        line = split_line_clean(line)
        if len(line) < 3:
            continue
        vertex_index = int(line[0])
        vertex_weights = []
        for i in range(1, len(line), 2):
            bone_index = int(line[i])
            bone_weight = float(line[i + 1])
            vertex_weights.append((bone_index, bone_weight))
        vertices_weights[vertex_index] = vertex_weights

    n_vertices = max(vertices_weights.keys()) + 1
    weight_matrix = np.zeros((n_vertices, n_bones))

    for vertex_index, vertex_weights in vertices_weights.items():
        for bone_index, bone_weight in vertex_weights:
            weight_matrix[vertex_index, bone_index] = bone_weight

    return bone_names, weight_matrix


# =================================== Obj ==================================

def read_obj(path: Path):
    vertices = []
    faces = []
    normals = []
    texture_coords = []
    text = path.read_text()
    for line in split_lines_and_remove_spaces(text):
        if line[0] == 'vn':
            normals.append([float(line[1]), float(line[2]), float(line[3])])
        if line[0] == 'v':
            vertices.append([float(line[1]), float(line[2]), float(line[3])])
        if line[0] == 'vt':
            texture_coords.append([float(line[1]), float(line[2])])
        if line[0] == 'f':
            v1, t1, vn1 = list(map(int, line[1].split('/')))
            v2, t2, vn2 = list(map(int, line[2].split('/')))
            v3, t3, vn3 = list(map(int, line[3].split('/')))
            faces.append([[v1, t1, vn1], [v2, t2, vn2], [v3, t3, vn3]])

    normals = np.array(normals)
    faces = np.array(faces, dtype=int)
    vertices = np.array(vertices)
    texture_coords = np.array(texture_coords)

    n_faces = faces.shape[0]
    n_vertices = vertices.shape[0]

    vertex_texture_coords = np.empty((n_vertices, 2))
    vertex_normals = np.empty((n_vertices, 3))

    for v, t, vn in faces.reshape((-1, 3)):
        vertex_texture_coords[v - 1] = texture_coords[t - 1]
        vertex_normals[v - 1] = normals[vn - 1]

    faces = faces[..., 0] - 1

    return {
        'n_vertices': n_vertices,
        'n_faces': n_faces,
        'vertices': vertices,
        'faces': faces,
        'vertex_normals': vertex_normals,
        'vertex_texture_coords': vertex_texture_coords,
    }

# =================================== Segmentation =========================


def read_segmentation(path: Path):
    text = path.read_text()
    vertex_labels = text.splitlines(keepends=False)
    vertex_labels = list(map(int, vertex_labels))
    return np.array(vertex_labels, dtype=np.uint8)


# ============================== Texture ===================================


def read_texture(texture_path: Path) -> ndarray:
    return np.array(Image.open(texture_path))


# ============================= Raw Model Data =============================


def read_livecap_model(obj_path: Path, skeleton_path: Path, skin_path: Path, segmentation_path: Path,
                       texture_map_path: Path, livecap_to_camera=np.identity(4),
                       scale_to_vibe: bool = True) -> RawModelData:

    obj_data = read_obj(obj_path)

    skinning_data = read_skinning_data(skin_path, skeleton_path)

    vertex_labels = read_segmentation(segmentation_path)

    texture = read_texture(texture_map_path)

    scale = 1
    if scale_to_vibe:
        joint_list = skinning_data['joint_list']
        joint_coords = np.array([joint.world_coordinates for joint in joint_list])
        # scale by the y axis
        ax = 1
        orig_scale = joint_coords.max(axis=0)[ax] - joint_coords.min(axis=0)[ax]
        scale = get_vibe_model_dimensions()[ax] / orig_scale

    return RawModelData(
        **obj_data,
        **skinning_data,
        n_keyframes=0,
        transformation_matrix=livecap_to_camera,
        vertex_labels=vertex_labels,
        texture=texture,
        scale=scale,
    )


def save_camera_params(src_calib_path, dataset_path, save_params_path):
    calibrations = read_calibration(src_calib_path)
    dataset = LiveCapDataset(root=dataset_path)
    w = dataset.image_width
    h = dataset.image_height
    chosen_calibration = None
    for calibration in calibrations:
        if calibration.width == w and calibration.height == h:
            chosen_calibration = calibration
            break
    if chosen_calibration is None:
        raise RuntimeError
    cam_params = chosen_calibration.intrinsic
    fx = cam_params[0, 0]
    fy = cam_params[1, 1]
    u = cam_params[0, 2]
    v = cam_params[1, 2]
    np.savez(save_params_path, fx=fx, fy=fy, u=u, v=v, h=h, w=w)


if __name__ == "__main__":
    import config
    save_camera_params(config.calibration_path, config.original_dataset_path, config.intrinsic_params_path)
