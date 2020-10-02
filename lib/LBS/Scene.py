import keyboard
import numpy as np
from scipy.spatial.transform import Rotation

from lib.LBS.Model import AnimatedModel
from lib.LBS.Animator import Animator


class Camera:
    def __init__(self, scale):
        self.position = [0, 0, 0]
        self.pitch = 0      # high/low      (x axis)
        self.yaw = 0        # left/right    (y axis)
        self.roll = 0       # tilted        (z axis)
        self.scale = scale

    def move(self):
        if keyboard.is_pressed('i'):
            # z in
            self.position[2] -= 0.01 * self.scale
        if keyboard.is_pressed('o'):
            # z out
            self.position[2] += 0.01 * self.scale

        if keyboard.is_pressed('w'):
            # y up
            self.position[1] += 0.01 * self.scale
        if keyboard.is_pressed('s'):
            # y down
            self.position[1] -= 0.01 * self.scale

        if keyboard.is_pressed('d'):
            # x right
            self.position[0] += 0.01 * self.scale
        if keyboard.is_pressed('a'):
            # x left
            self.position[0] -= 0.01 * self.scale

        if keyboard.is_pressed('up'):
            self.pitch += 0.08 * self.scale
        if keyboard.is_pressed('down'):
            self.pitch -= 0.08 * self.scale
        if keyboard.is_pressed('left'):
            self.yaw += 0.08 * self.scale
        if keyboard.is_pressed('right'):
            self.yaw -= 0.08 * self.scale


class Entity:
    def __init__(self, animated_model: AnimatedModel, position=[0, 0, 0], rotation=[0, 0, 0], scale=1.0):
        assert len(position) == 3 and len(rotation) == 3, "position, rotation should contain exactly 3 elements"
        self.animated_model: AnimatedModel = animated_model
        self.position = position
        self.rotation = rotation
        self.scale = scale

        self.animator = Animator(self.animated_model)
        return

    def increase_position(self, position_delta):
        assert len(position_delta) == 3, "position_delta should contain exactly 3 elements"
        self.position = [sum(x) for x in zip(self.position, position_delta)]

    def increase_rotation(self, rotation_delta):
        assert len(rotation_delta) == 3, "rotation_delta should contain exactly 3 elements"
        self.rotation = [sum(x) for x in zip(self.rotation, rotation_delta)]

    def update(self):
        self.animator.update()


class Scene:
    def __init__(self, entity: Entity, camera: Camera):
        self.entity: Entity = entity
        self.camera = camera
        self.lightDirection = None
        return


def create_view_matrix(camera: Camera) -> np.ndarray:
    view_matrix: np.ndarray = np.eye(4)
    rotation = Rotation.from_euler('xyz', [camera.pitch, camera.yaw, camera.roll], degrees=True).as_matrix()
    view_matrix[:3, :3] = rotation
    view_matrix[:3, 3] = [-x for x in camera.position]
    return view_matrix
