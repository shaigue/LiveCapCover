"""Helpful code for creating the mapping from vibe joints names to model joints names

you do this manually using blender at the same time
"""

from pathlib import Path
from pprint import pformat
from typing import Union, List, Dict
from dataclasses import dataclass

from matplotlib import pyplot as plt
import numpy as np
from numpy import ndarray

import config
from lib.data_utils.livecap_dataset import LiveCapDataset, LiveCapEntry
from lib.image_processing.vibe.lib.data_utils.kp_utils import get_spin_joint_names
from lib.image_processing.vibe.lib.models.smpl import SMPL, SMPL_MODEL_DIR
from lib.utils.model import Model


def get_vibe_to_blender_joints() -> Dict[str, str]:
    """Returns:
        kp_to_joint_map(Dict[str, str]): a mapping between keypoint name in vibe to a joint name in the model.
        Some joints are not pointed to, and some keypoints do not match a joint(None value)
    """
    return {
        'Head (H36M)': None,
        'Jaw (H36M)': 'lower_lip',
        'OP LAnkle': 'foot_L',
        'OP LBigToe': 'toe_L',
        'OP LEar': None,
        'OP LElbow': 'forearm_L',
        'OP LEye': 'left_eye',
        'OP LHeel': 'heel_02_L',
        'OP LHip': 'pelvis_L',
        'OP LKnee': 'shin_L',
        'OP LShoulder': 'upper_arm_L',
        'OP LSmallToe': None,
        'OP LWrist': 'hand_L',
        'OP MidHip': 'spine',
        'OP Neck': 'spine_005',
        'OP Nose': 'nose',
        'OP RAnkle': 'foot_R',
        'OP RBigToe': 'toe_R',
        'OP REar': None,
        'OP RElbow': 'forearm_R',
        'OP REye': 'right_eye',
        'OP RHeel': 'heel_02_R',
        'OP RHip': 'pelvis_R',
        'OP RKnee': 'shin_R',
        'OP RShoulder': 'upper_arm_R',
        'OP RSmallToe': None,
        'OP RWrist': 'hand_R',
        'Spine (H36M)': 'spine_002',
        'headtop': 'spine_006',
        'hip': None,
        'lankle': None,
        'lear': None,
        'lelbow': None,
        'leye': None,
        'lhip': 'thigh_L',
        'lknee': None,
        'lshoulder': None,
        'lwrist': None,
        'neck': None,
        'nose': None,
        'rankle': None,
        'rear': None,
        'relbow': None,
        'reye': None,
        'rhip': 'thigh_R',
        'rknee': None,
        'rshoulder': None,
        'rwrist': None,
        'thorax': 'spine_004'
    }


def get_vibe_to_original_joints() -> Dict[str, str]:
    """Returns:
        kp_to_joint_map(Dict[str, str]): a mapping between keypoint name in vibe to a joint name in the model.
        Some joints are not pointed to, and some keypoints do not match a joint(None value)
    """
    return {
        'Head (H36M)': 'head_ee_ry',
        'Jaw (H36M)': 'chin',
        'OP LAnkle': 'left_ankle_ry',
        'OP LBigToe': 'left_foot_ee',
        'OP LEar': None,
        'OP LElbow': 'left_elbow_rx',
        'OP LEye': 'lefteye',
        'OP LHeel': None,
        'OP LHip': 'left_hip_rx',
        'OP LKnee': 'left_knee_rx',
        'OP LShoulder': 'left_shoulder_ry',
        'OP LSmallToe': None,
        'OP LWrist': 'left_hand_ry',
        'OP MidHip': None,
        'OP Neck': 'neck_1_rx',
        'OP Nose': 'nose',
        'OP RAnkle': 'right_ankle_ry',
        'OP RBigToe': 'right_foot_ee',
        'OP REar': None,
        'OP RElbow': 'right_elbow_rx',
        'OP REye': 'righteye',
        'OP RHeel': None,
        'OP RHip': 'right_hip_rx',
        'OP RKnee': 'right_knee_rx',
        'OP RShoulder': 'right_shoulder_ry',
        'OP RSmallToe': None,
        'OP RWrist': 'right_hand_ry',
        'Spine (H36M)': 'spine_1_ry',
        'headtop': None,
        'hip': None,
        'lankle': None,
        'lear': None,
        'lelbow': None,
        'leye': None,
        'lhip': None,
        'lknee': None,
        'lshoulder': None,
        'lwrist': None,
        'neck': None,
        'nose': None,
        'rankle': None,
        'rear': None,
        'relbow': None,
        'reye': None,
        'rhip': None,
        'rknee': None,
        'rshoulder': None,
        'rwrist': None,
        'thorax': None,
    }


@dataclass
class AdapterEntry:
    frame: ndarray
    kp_2d: ndarray
    kp_3d: ndarray
    kp_3d_translation: ndarray
    silhouette: ndarray


def get_vibe_model_dimensions():
    """Returns the scale on each axis of the model's bounding box."""
    smpl = SMPL(SMPL_MODEL_DIR, create_transl=False)
    rest_pose = smpl()
    kp_3d = rest_pose.joints.squeeze().detach().numpy()

    dimensions = kp_3d.max(axis=0) - kp_3d.min(axis=0)

    return dimensions


class LiveCapAdapter:
    """Class that adapts the LiveCapDataset into our desired form of output.

    Attributes:
        ds(LiveCapDataset): internal livecap dataset
        joint_indices(List[int]): list of joint indices that correspond to the keypoints
        kp_indices(List[int]): list of keypoint indices that correspond to the joints, s.t. the joint_indices[i] is the
            index of the joint that matches the keypoint index kp_indices[i]

    """

    def __init__(self, root: Union[str, Path], model: Model, camera_fitting_path: Union[str, Path],
                 kp_to_joint_map=None):
        """

        Arguments:
            root: path to the root of the dataset
            model: the model that we fit to.
            camera_fitting_path: path to the file containing fx, fy, translations for each frame
        """
        if kp_to_joint_map is None:
            kp_to_joint_map = get_vibe_to_blender_joints()
        self.ds = LiveCapDataset(root)
        self.n = self.ds.n
        self.joint_indices, self.kp_indices = self._get_indices(model, kp_to_joint_map)
        self.image_h = self.ds.image_height
        self.image_w = self.ds.image_width
        camera_fitting_params = np.load(camera_fitting_path)
        self.fx = camera_fitting_params['fx']
        self.fy = camera_fitting_params['fy']
        self.translation = camera_fitting_params['t']
        self._smooth_translation()

    def _smooth_translation(self):
        smooth_translation = np.empty_like(self.translation)
        n = len(self.translation)
        for i in range(n):
            if i == 0 or i == (n - 1):
                smooth_translation[i] = self.translation[i]
            else:
                smooth_translation[i] = 0.3 * self.translation[i-1] + 0.4 * self.translation[i] \
                                        + 0.3 * self.translation[i+1]
        self.translation = smooth_translation

    def get_initial_translation(self):
        return self.translation[0]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i) -> AdapterEntry:
        livecap_entry = self.ds[i]
        frame = livecap_entry.frame
        silhouette = livecap_entry.silhouette
        kp_2d = self._process_kp_2d(livecap_entry)
        kp_3d = self._process_kp_3d(livecap_entry)
        kp_3d_translation = self.translation[i]

        return AdapterEntry(frame, kp_2d, kp_3d, kp_3d_translation, silhouette)

    def _process_kp_2d(self, livecap_entry: LiveCapEntry) -> ndarray:
        kp_2d = livecap_entry.vibe.kp_2d
        # take only the keypoint that we are interested in
        return kp_2d[self.kp_indices]

    def _process_kp_3d(self, livecap_entry: LiveCapEntry) -> ndarray:
        # scale and add translation
        kp_3d = livecap_entry.vibe.kp_3d[self.kp_indices]
        return kp_3d

    @staticmethod
    def _get_indices(model: Model, kp_to_joint_map: dict) -> (ndarray, ndarray):
        vibe_kp_names = get_spin_joint_names()

        joint_indices = []
        kp_indices = []
        for kp_index, kp_name in enumerate(vibe_kp_names):
            if kp_name not in kp_to_joint_map:
                raise RuntimeError("All vibe Keypoints should be in the map.")
            joint_name = kp_to_joint_map[kp_name]
            if joint_name is None:
                continue

            if model.face_parts_list is not None and joint_name in model.face_parts_list:
                # if the joint_name got is a face part, then it will be concatenated after all of the joints
                # in the same order it appears in face_parts_list
                joint_index = model.n_joints + model.face_parts_list.index(joint_name)
            else:
                joint = model.root_joint.find_joint_by_name(joint_name)
                if joint is None:
                    raise RuntimeError("Found a joint name that is not in the skeleton. make sure to use the correct"
                                       "model.")
                joint_index = joint.index
            joint_indices.append(joint_index)
            kp_indices.append(kp_index)

        joint_indices = np.array(joint_indices)
        kp_indices = np.array(kp_indices)

        return joint_indices, kp_indices

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        if self._i < self.n:
            item = self[self._i]
            self._i = self._i + 1
            return item
        else:
            raise StopIteration


def write_vibe_joints_names_as_dict():
    """Write all of the vibe joints names to a file, as a dict."""
    vibe_joints_names = get_spin_joint_names()
    vibe_joints_dict = {name: None for name in vibe_joints_names}
    vibe_joints_names_file = Path(__file__).parent / 'vibe_joints_names'
    with vibe_joints_names_file.open('w') as f:
        f.write(pformat(vibe_joints_dict))


def draw_vibe_joint_one_by_one():
    """Draw the joints one by one on an image"""
    vibe_joints_name = get_spin_joint_names()
    ds = LiveCapDataset(config.livecap_dataset_path)
    i = 800
    entry = ds[i]
    for j, name in enumerate(vibe_joints_name):
        print('kp name: ', name)
        kp_2d = entry.vibe.kp_2d[j]
        plt.imshow(entry.frame)
        plt.scatter(kp_2d[1], kp_2d[0])
        plt.show()
