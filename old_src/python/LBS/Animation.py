from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform.rotation import Rotation
from scipy.spatial.transform import Slerp
import numpy as np


class KeyFrame:
    def __init__(self, timestamp: float, joints_transforms: list):
        '''
        self.timestamp:     the time in seconds from the start of the animation when this key frame occurs.
        self.pose:          a map/dictionary. Each entry has the name of the joint and the bone-space transform of this joint (in relation
                            to parent joint) at self.timestamp.
        '''
        self.timestamp = timestamp
        self.pose = joints_transforms
        return


class Animation:
    def __init__(self, key_frames):
        # TODO get the right length
        self.key_frames = key_frames
        self.length = key_frames[-1].timestamp      # length of the entire animation, in seconds

        if len(key_frames) == 1:
            self.key_frames.append(key_frames[0])
        return


def interpolate_transformations(tr1: np.array, tr1_time: float, tr2: np.array, tr2_time: float, current_time: float):
    '''
    '''
    assert tr1.shape == tr2.shape == (4, 4)
    assert tr1_time <= current_time <= tr2_time

    # extract the rotation and position of each translation
    rotation1 = tr1[:3, :3]
    rotation2 = tr2[:3, :3]
    position1 = tr1[:3, 3]
    position2 = tr2[:3, 3]

    # interpolate rotation
    key_rotations: Rotation = R.from_matrix((rotation1, rotation2))
    key_times = [tr1_time, tr2_time]
    slerp = Slerp(key_times, key_rotations)
    interpolated_rotation = slerp([current_time]).as_matrix()[0]

    # interpolate position

    p = (current_time - tr1_time) / (tr2_time - tr1_time)
    interpolated_position = (1-p)*position1 + p*position2

    # assemble the interpolated translation and return it
    interpolated_translation = np.eye(4)
    interpolated_translation[:3, :3] = interpolated_rotation
    interpolated_translation[:3, 3] = interpolated_position
    return interpolated_translation


def interpolate_key_frames(key_frame1: KeyFrame, key_frame2: KeyFrame, current_time: float):
    pose1 = key_frame1.pose
    pose2 = key_frame2.pose
    time1 = key_frame1.timestamp
    time2 = key_frame2.timestamp
    interpolated_pose = []
    for i in range(len(key_frame1.pose)):
        interpolated_rotation = interpolate_transformations(pose1[i], time1, pose2[i], time2, current_time)
        interpolated_pose.append(interpolated_rotation)
    return interpolated_pose
