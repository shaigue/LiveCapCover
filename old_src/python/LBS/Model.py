import numpy as np
from Joint import Joint
# from Animator import Animator


class RawModel:
    def __init__(self, vaoID, vertex_count):
        self.vaoID = vaoID
        self.vertex_count = int(vertex_count)


class AnimatedModel:
    def __init__(self, raw_model: RawModel, textureID: int, root_joint: Joint, num_joints: int):
        # skin
        self.raw_model = raw_model
        self.textureID = textureID

        # skeleton
        self.root_joint = root_joint
        self.num_joints = num_joints
        assert len(root_joint) == num_joints
        return

    # def delete(self):
    #     self.model.delete()
    #     self.texture.delete()

    # def do_animation(self):
    #     self.animator.do_animation()
    #     return

    # def update(self):
    #     self.animator.update()
    #     return

    def __add_joints_to_array(self, head_joint: Joint, joints_transforms: np.array):
        joints_transforms[head_joint.index] = head_joint.joint_transform
        for child_joint in head_joint.children:
            self.__add_joints_to_array(child_joint, joints_transforms)

    def get_joints_transforms(self):
        joints_transforms = np.zeros((self.num_joints, 4, 4))
        self.__add_joints_to_array(self.root_joint, joints_transforms)
        return joints_transforms
