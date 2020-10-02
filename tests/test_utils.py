import unittest

import numpy as np

import config
from lib.IO.read_collada import read_collada_file
from lib.utils.utils import array_elements_in_range


class TestDataStructures(unittest.TestCase):
    def setUp(self) -> None:
        self.raw_model_data = read_collada_file(config.animated_model_path)

    def test_reduce_weight_matrix(self):
        vertex_joints, vertex_joints_weights = self.raw_model_data.reduce_weight_matrix()
        # check shapes
        self.assertEqual(vertex_joints.shape, (self.raw_model_data.n_vertices, 3))
        self.assertEqual(vertex_joints_weights.shape, (self.raw_model_data.n_vertices, 3))
        # check that weights sum up to one
        total_weights = vertex_joints_weights.sum(axis=1)
        self.assertTrue(np.allclose(total_weights, 1))
        # check that joint indices are in the correct bounds
        self.assertTrue(array_elements_in_range(vertex_joints, 0, self.raw_model_data.n_bound_joints))
        # print("weights: \n", raw_model_data.weight_matrix[:5])
        # print("joints: \n", vertex_joints[:5])
        # print("new weights: \n", vertex_joints_weights[:5])

    def test_preorder_iterator(self):
        n = 0
        for joint in self.raw_model_data.root_joint.preorder_iterator():
            n += 1
            # print(joint)
        self.assertEqual(n, self.raw_model_data.n_total_joints)

    def test_setup_animation(self):
        animation = self.raw_model_data.get_animation()
        self.assertGreater(animation.length, 0)
        self.assertEqual(len(animation.key_frames), self.raw_model_data.n_keyframes)
        for keyframe in animation.key_frames:
            self.assertGreater(keyframe.timestamp, 0)
            self.assertLessEqual(keyframe.timestamp, animation.length)
            self.assertEqual(self.raw_model_data.n_total_joints, len(keyframe.pose))


if __name__ == '__main__':
    unittest.main()
