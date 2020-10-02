from Animation import Animation, interpolate_key_frames
from Model import AnimatedModel
from Joint import Joint
from WindowManager import WindowManager

import numpy as np


class Animator:
    def __init__(self, animated_model: AnimatedModel):
        self.animated_model = animated_model
        self.animation_time = 0.0
        self.animation: Animation = None
        self.prev_key_frame_idx = None

    def set_animation(self, animation: Animation):
        '''
            Indicates that the entity should carry out the given animation. Resets the animation time so that the new animation starts from
            the beginning.
        '''
        self.animation = animation
        self.animation_time = 0.0
        self.prev_key_frame_idx = 0
        WindowManager.set_time(0.0)

    def update(self):
        if not self.animation:
            return

        self.__increase_animation_time()
        current_pose = self.__calculate_current_animation_pose()
        self.__apply_pose_to_joints(current_pose, self.animated_model.root_joint, np.eye(4))

    def __increase_animation_time(self):
        self.animation_time += WindowManager.get_current_time()
        if self.animation_time > self.animation.length:
            self.set_animation(self.animation)
        elif self.animation_time > self.animation.key_frames[self.prev_key_frame_idx+1].timestamp:
            self.prev_key_frame_idx += 1
        return

    def __calculate_current_animation_pose(self):
        prev_key_frame = self.animation.key_frames[self.prev_key_frame_idx]
        next_key_frame = self.animation.key_frames[self.prev_key_frame_idx+1]
        current_pose = interpolate_key_frames(prev_key_frame, next_key_frame, self.animation_time)
        return current_pose

    def __apply_pose_to_joints(self, current_pose, joint: Joint, parent_current_transform: np.array):
        current_local_transform = current_pose[joint.index]
        current_transform = np.dot(parent_current_transform, current_local_transform)
        for child in joint.children:
            self.__apply_pose_to_joints(current_pose, child, current_transform)
        joint.joint_transform = np.dot(current_transform, joint.inverse_bind_transform)
        return
