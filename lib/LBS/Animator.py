from lib.LBS.Animation import Animation, interpolate_key_frames
from lib.LBS.Model import AnimatedModel
from lib.LBS.WindowManager import WindowManager

import numpy as np


class Animator:
    def __init__(self, animated_model: AnimatedModel):
        self.animated_model: AnimatedModel = animated_model
        self.animation_time = None
        self.animation: Animation = None
        self.prev_key_frame_idx = None

    def set_animation(self, animation: Animation):
        '''
            Indicates that the entity should carry out the given animation. Resets the animation time so that the new animation starts from
            the beginning.
        '''
        if animation is None:
            return
        self.animation = animation
        # self.animation_time = 0.0
        self.animation_time = self.animation.key_frames[0].timestamp
        self.prev_key_frame_idx = 0
        WindowManager.set_time(self.animation_time)

    def update(self):
        if self.animation is None:
            return

        self._increase_animation_time()
        new_pose = self._calculate_new_animation_pose()
        self.animated_model.root_joint.apply_pose_to_joints(new_pose)

    def _increase_animation_time(self):
        self.animation_time += WindowManager.get_current_time()
        # self.animation_time += 0.04
        if self.animation_time > self.animation.length:
            self.set_animation(self.animation)
        while self.animation_time > self.animation.key_frames[self.prev_key_frame_idx+1].timestamp:
            self.prev_key_frame_idx += 1
            assert self.prev_key_frame_idx < len(self.animation.key_frames)
        return

    def _calculate_new_animation_pose(self):
        prev_key_frame = self.animation.key_frames[self.prev_key_frame_idx]
        next_key_frame = self.animation.key_frames[self.prev_key_frame_idx+1]
        new_pose = interpolate_key_frames(prev_key_frame, next_key_frame, self.animation_time)
        return new_pose
