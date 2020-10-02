"""module for representing a joint."""

from typing import List
from enum import Enum, auto
from numpy import ndarray

from lib.utils.transformation import Transformation


class JointAnimation:
    """Class that represents an animation for a single joint.

    Attributes:
        timestamps(ndarray)[shape=(n_keyframes, ), dtype=float]: keyframe timestamps
        poses(ndarray)[shape=(n_keyframes, 4, 4), dtype=float]: keyframe poses
    """

    def __init__(self):
        self.timestamps: ndarray = None
        self.poses: ndarray = None


class MovementType(Enum):
    translation = auto()
    rotation = auto()


class Joint:
    """Represents a Joint.

    Attributes:
        name(str): name of the joint
        children(List[Joint]): a list of all of the joint's children
        is_root(bool): if this joint is the root of the joint tree.
        index(int): an index property that is in use in the model's rendering
        t_joint_to_parent(ndarray): transformation from the joint to it's parent.
            corresponds to the 'matrix' attribute in the collada.Node
        t_model_to_joint(ndarray): transformation from the model space to the joint space, also
            called the 'inverse_bind_matrix'
        t_model_to_world(ndarray): the cumulative transformation that a joint operates on a joint. i.e., in run time,
            to get the vertex location with respect to that joint we only multiply by this matrix.
            also called 'joint_matrix'
        animation(JointAnimation): this is a JointAnimation object that is associated with this joint

    Methods:
        __init__: creates a new Joint
        set_root: sets the Joint as the root of the joint tree
        init_t_model_to_joint: initializes the 't_model_to_joint' of all the joints in the tree,
            this method should only be called on the root of the tree for the results to be correct
        calc_joint_matrices: calculates the 't_model_to_world' matrix for all the joints in the tree.
            this method should only be called on the root
        print_tree: prints the joint tree

    Notes:
        reference frames:
            model - where the vertex coordinates are
            joint - joint local coordinate system where the joint is at the origin
            parent - the joint parent's local coordinate system
            world - the coordinate system in the world
    """

    def __init__(self, name: str, t_joint_to_parent: ndarray, index: int = None,
                 # This is for supporting livecap models
                 axis: ndarray = None,
                 movement_type: MovementType = None,
                 ):
        if not isinstance(t_joint_to_parent, ndarray):
            raise ValueError
        if t_joint_to_parent.shape != (4, 4):
            raise ValueError

        self.is_root = False
        self.index = index
        self.name = name
        self.children:  List[Joint] = []

        self.t_initial_joint_to_parent = Transformation(t_joint_to_parent.copy())  # initial transformation from the parent
        self.t_current_joint_to_parent = Transformation(t_joint_to_parent.copy())  # current transformation from the parent
        self.t_model_to_joint: Transformation = None                            # inverse bind transform

        self.t_joint = Transformation()                     # this matrix will be applied to vertices
        self.t_joint_to_world = Transformation()            # this is in order to get the current position

        self.animation = JointAnimation()

        self.axis = axis
        self.movement_type = movement_type

    def set_root(self):
        self.is_root = True

    def init_skeleton(self):
        if not self.is_root:
            raise RuntimeError("This method should only be called on the joint root")
        self._init_t_model_to_joint()
        self.calc_t_joint()

    def calc_t_joint(self, t_parent_to_world: Transformation = None):
        if t_parent_to_world is None:
            t_parent_to_world = Transformation()
        self.t_joint_to_world = t_parent_to_world.compose(self.t_current_joint_to_parent)
        self.t_joint = self.t_joint_to_world.compose(self.t_model_to_joint)

        for child in self.children:
            child.calc_t_joint(self.t_joint_to_world)

    def _init_t_model_to_joint(self, t_parent_to_model: Transformation = None):
        if t_parent_to_model is None:
            t_parent_to_model = Transformation()

        t_joint_to_model = t_parent_to_model.compose(self.t_initial_joint_to_parent)
        self.t_model_to_joint = t_joint_to_model.inverse()
        for child in self.children:
            child._init_t_model_to_joint(t_joint_to_model)

    @property
    def initial_translation(self):
        return self.t_initial_joint_to_parent.translation

    @property
    def initial_angles(self):
        return self.t_initial_joint_to_parent.angles

    @property
    def world_coordinates(self):
        return self.t_joint_to_world.translation

    def apply_along_axis(self, value: float):
        if self.axis is None or self.movement_type is None:
            raise RuntimeError
        vector = self.axis * value
        if self.movement_type == MovementType.translation:
            self.t_current_joint_to_parent.translation = vector
        elif self.movement_type == MovementType.rotation:
            self.t_current_joint_to_parent.rotvec = vector

    def set_joint_to_parent_angles(self, angles: ndarray):
        self.t_current_joint_to_parent.angles = angles

    def set_joint_to_parent_translation(self, translation: ndarray):
        self.t_current_joint_to_parent.translation = translation

    def convert_basis(self, t_old_to_new: Transformation, t_new_to_old: Transformation = None, scale: float = 1):
        self.t_initial_joint_to_parent.convert_basis(t_old_to_new, t_new_to_old, scale)
        self.t_current_joint_to_parent.convert_basis(t_old_to_new, t_new_to_old, scale)
        self.t_model_to_joint.convert_basis(t_old_to_new, t_new_to_old, scale)
        self.t_joint.convert_basis(t_old_to_new, t_new_to_old, scale)
        self.t_joint_to_world.convert_basis(t_old_to_new, t_new_to_old, scale)
        if self.axis is not None:
            self.axis = t_old_to_new.apply(self.axis, is_point=False)

    def find_joint_by_name(self, name: str):
        """Finds a joint in this joint's tree by its name."""
        if self.name == name:
            return self
        for child in self.children:
            joint = child.find_joint_by_name(name)
            if joint is not None:
                return joint
        return None

    def find_joint_by_index(self, index: int):
        if self.index == index:
            return self
        for child in self.children:
            joint = child.find_joint_by_index(index)
            if joint is not None:
                return joint

    def has_animation(self) -> bool:
        return self.animation.poses is not None and self.animation.timestamps is not None

    def __str__(self):
        return f'<Joint: index={self.index}, name=\'{self.name}\', n_children={len(self.children)}, len={len(self)}>'

    def __len__(self):
        n_joints = 1
        for child in self.children:
            n_joints += len(child)
        return n_joints

    def print_tree(self, level: int = 0):
        print('\t' * level, self)
        for child in self.children:
            child.print_tree(level + 1)

    def preorder_iterator(self):
        yield self
        for child in self.children:
            yield from child.preorder_iterator()
