import numpy as np
from lib.utils.raw_model_data import Joint
from lib.LBS.Animation import Animation
from lib.utils.raw_model_data import RawModelData


class RawModel:
    def __init__(self, vaoID, vertex_count):
        self.vaoID = vaoID
        self.vertex_count = int(vertex_count)


class AnimatedModel:
    def __init__(self, raw_model: RawModel, textureID: int, root_joint: Joint):
        # skin
        self.raw_model = raw_model
        self.textureID = textureID

        # skeleton
        self.root_joint: Joint = root_joint
        self.num_joints = len(root_joint)
        return

    def __add_joints_to_array(self, head_joint: Joint, joints_transforms: np.array):
        joints_transforms[head_joint.index] = head_joint.t_model_to_world
        for child_joint in head_joint.children:
            self.__add_joints_to_array(child_joint, joints_transforms)

    def get_joints_transforms(self) -> np.ndarray:
        joints_transforms: np.ndarray = np.zeros((self.num_joints, 4, 4))
        self.__add_joints_to_array(self.root_joint, joints_transforms)
        return joints_transforms


class Model():
    ''' This is the resulting data structure that the reader will produce,
    will be used by the model's init function.

    Attributes:
        n_vertices - number of vertices in the model
        n_faces - number of polygons in the model
        Model:
            faces (numpy.ndarray) [dtype=int32, shape=(n_faces, 3)]:
                indices that define the model's polygons (triangles). faces.max()+1 == n_vertices
            vertices (numpy.ndarray) [dtype=float32, shape=(n_vertices, 3)]:
                vertices[i] are the coordinates of vertex i.
            vertex_texture_coords (numpy.ndarray) [dtype=float32, shape=(n_vertices, 2)]:
                vertex_texture_coords[i] are the texture coordinates of vertex i.
            vertex_normals (numpy.ndarray) [dtype=float32, shape=(n_vertices, 3)]:
                vertex_normals[i] are the normal coordinates of vertex i.

        Skeleton:
            vertex_joints (numpy.ndarray) [dtype=int32, shape=(n_vertices,3)]:
                vertex_joints[i] are the indices of the three joints that affect vertex i the most (linear blend skinning).
            vertex_joints_weights (numpy.ndarray) [dtype=float32, shape=(n_vertices,3)]:
                vertex_joints_weights[i] are the vertex_joints_weights of vertex_joints[i], respectively. Must sum up to 1.
            root_joint (Joint)

        Animation:
            key_frames (list[KeyFrame])

        assert self.faces.max()+1 == len(self.vertices) == len(self.vertex_texture_coords) == len(
            self.vertex_normals) == len(self.vertex_joints) == len(self.vertex_joints_weights)
    '''

    def __init__(self, faces, vertices, vertex_texture_coords, vertex_normals, root_joint: Joint, vertex_joints, vertex_joints_weights,
                 animation: Animation):
        # mesh
        self.faces = faces.astype('int32')
        self.vertices = vertices.astype('float32')
        self.vertex_texture_coords = vertex_texture_coords.astype('float32')
        self.vertex_normals = vertex_normals.astype('float32')

        # skeleton
        self.root_joint: Joint = root_joint

        # skinning
        self.vertex_joints = vertex_joints.astype('int32')
        self.vertex_joints_weights = vertex_joints_weights.astype('float32')
        assert self.faces.max()+1 == len(self.vertices) == len(self.vertex_texture_coords) == len(
            self.vertex_normals) == len(self.vertex_joints) == len(self.vertex_joints_weights)

        # animation
        self.animation = animation

        self.root_joint.print_tree()
        return

    @classmethod
    def from_raw_model_data(cls, raw_model_data: RawModelData):
        vertex_joints, vertex_joints_weights = raw_model_data.reduce_weight_matrix()
        animation = raw_model_data.get_animation()
        return cls(faces=raw_model_data.faces, vertices=raw_model_data.vertices, vertex_texture_coords=raw_model_data.vertex_texture_coords,
                   vertex_normals=raw_model_data.vertex_normals, root_joint=raw_model_data.root_joint, vertex_joints=vertex_joints,
                   vertex_joints_weights=vertex_joints_weights, animation=animation)
