from Model import AnimatedModel
from Animator import Animator


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
