import numpy as np


class Joint:
    def __init__(self, index, name, local_bind_transform, inverse_bind_transform):
        '''
        self.index - the index of this joint in the uniform array in the shader.

        self.children - the joint's children joints.

        self.local_bind_transform - original transform of the joint in relation to its parent joint. Used in the calculation of the joints
        transforms in the animator class (it is actually only used to calculate self.inverse_bind_transform).

        self.inverse_bind_transform - the inverse bind transform of the joint, which means it transforms the joint from its original
        position in the model to the model's origin. used in the calculation of the joint's transforms in the animator.

        self.joint_transform - the tranform needed to move the joint from its original position in the model (when no animation at all
        is applied) to the position in the current pose. the transform is in the model space. This matrix transform gets loaded up to the
        shader in the uniform array and stored at self.index position in that array.
        '''

        self.index = index                  # ID
        self.name = name
        self.children = []

        assert local_bind_transform.shape == inverse_bind_transform.shape == (4, 4)
        self.local_bind_transform: np.array = local_bind_transform
        self.inverse_bind_transform: np.array = inverse_bind_transform
        self._inverse_bind_transform = None         # for testing, see _test_inverse_bind_transform

        self.joint_transform = np.eye(4)            # joint matrix

    def __len__(self):
        size = 1
        for child in self.children:
            size += len(child)
        return size

    def __str__(self, level=0):
        ret = "\t"*level+repr(self.name)+"\n"
        for child in self.children:
            ret += child.__str__(level+1)
        return ret

    def __calculate_inverse_bind_transform(self, parent_bind_transform: np.array):
        '''
        self.local_bind_transform is the joint's bind transform relative to its parent. parent_bind_transform is the parent's world bind
        transform. Therefore if we multiply both we get the joint's world bind transform. Then inversing it would get us the inverse bind
        transform.
        '''
        assert parent_bind_transform.shape == (4, 4)
        bind_transform = np.dot(parent_bind_transform, self.local_bind_transform)
        self._inverse_bind_transform = np.linalg.inv(bind_transform)
        for child in self.children:
            child.__calculate_inverse_bind_transform(bind_transform)

    def __test_inverse_bind_transform(self):
        '''
        Recursively compare self.inverse_bind_transform (what we got from the file) to self._inverse_bind_transform (what we calculated).
        This test method assumes that __calculate_inverse_bind_transform was called on the root joint.
        '''
        assert np.allclose(self.inverse_bind_transform, self._inverse_bind_transform, atol=0.0001)
        for child in self.children:
            child.__test_inverse_bind_transform()

    def _test_inverse_bind_transform(self):
        '''
        Call this test method on the root joint only!
        This is a test method with a purpose to make sure that the inverse_bind_transform that we extracted from the collada file makes
        sense.
        First we run __calculate_inverse_bind_transform to recursively calculate _inverse_bind_transform matrix out of the joint's
        parent_bind_transform and its local_bind_transform. Then we compare the calculation to the inverse_bind_transform that we got
        from the collada file and make sure that both are equal (within a margin error).
        '''
        self.__calculate_inverse_bind_transform(np.eye(4))
        self.__test_inverse_bind_transform()

    def add_child(self, child):
        self.children.append(child)
