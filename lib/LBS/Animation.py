from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform.rotation import Rotation
from scipy.spatial.transform import Slerp
import numpy as np
from typing import List

from lib.utils.transformation import Transformation


class KeyFrame:
    def __init__(self, joints_transforms: List[Transformation], timestamp: float):
        '''
        self.pose:          a list of Transformation objects. Each entry has the name of the joint and the bone-space transform of this joint
                            (in relation to parent joint) at self.timestamp.
        self.timestamp:     the time in seconds from the start of the animation when this key frame occurs.
        '''
        self.timestamp = timestamp
        self.pose: List[Transformation] = joints_transforms
        return


class Animation:
    def __init__(self, key_frames):
        # TODO get the right length
        self.key_frames: List[KeyFrame] = key_frames
        self.length: float = key_frames[-1].timestamp      # length of the entire animation, in seconds

        if len(key_frames) == 1:
            self.key_frames.append(key_frames[0])
        return


def interpolate_transformations(tr1: Transformation, tr1_time: float, tr2: Transformation, tr2_time: float, current_time: float) -> Transformation:
    '''
    '''
    assert tr1_time <= current_time <= tr2_time

    # interpolate rotations
    key_times = [tr1_time, tr2_time]
    key_rotations: Rotation = R.from_matrix((tr1.rotation.as_matrix(), tr2.rotation.as_matrix()))
    slerp = Slerp(key_times, key_rotations)
    interpolated_rotation = slerp([current_time]).as_matrix()[0]

    # interpolate translation
    p = (current_time - tr1_time) / (tr2_time - tr1_time)
    interpolated_translation = (1-p)*tr1.translation + p*tr2.translation

    # return the interpolated transformation
    interpolated_transform = Transformation(R.from_matrix(interpolated_rotation), interpolated_translation)
    return interpolated_transform


def interpolate_key_frames(key_frame1: KeyFrame, key_frame2: KeyFrame, current_time: float) -> List[Transformation]:
    pose1 = key_frame1.pose
    pose2 = key_frame2.pose
    time1 = key_frame1.timestamp
    time2 = key_frame2.timestamp
    interpolated_pose: List[Transformation] = []
    for i in range(len(pose1)):
        interpolated_transform: Transformation = interpolate_transformations(pose1[i], time1, pose2[i], time2, current_time)
        interpolated_pose.append(interpolated_transform)
    return interpolated_pose
