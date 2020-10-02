from Joint import Joint
from Animation import KeyFrame

import collada
import numpy as np
import trimesh
from os.path import splitext


# vertices, indices, texture_coords = load_obj_mesh(model_path)

class ColladaParser:
    '''
    When parsing a collada file we need to extract data about two main parts of the model - the mesh and the skeleton. We also need to
    extract data about the animation

    The mesh data includes the vertices, texture_coords and normals. It also includes the indices.
    The skeleton data includes information about the skeleton of the model - the joints, their hierarchy, transformation matrices (local
    bind transform, inverse bind transform); also, for each vertex we have the joints that affect him and the respective weights (used for LBS).
    The animation data mainly includes the key frames.

    Basically this function extracts vertices and texture data. It is needed because
    For comparison, when load_collada_model_data_no_texture extracts vertices and indices
    vertices = triset.vertex.flatten('C')
    indices = triset.vertex_index.flatten('C')
    '''

    def __init__(self, model_path):
        ext = splitext(model_path)[1].lower()
        if ext != '.dae':
            print("Please use only .dae files. Exiting...")
            raise SystemExit

        self.model_data = collada.Collada(model_path)
        # vertices, indices, texture_coords = self.__parse_model_data()
        indices, vertices, texture_coords, vertex_joints, weights, root_joint, key_frames = self.__parse_model_data_no_texture()

        normals = np.zeros(len(vertices))
        self.indices = indices.astype('int32')
        self.vertices = vertices.astype('float32')
        self.texture_coords = texture_coords.astype('float32')
        self.normals = normals.astype('float32')
        self.vertex_joints = vertex_joints.astype('int32')
        self.weights = weights.astype('float32')
        assert self.indices.max()+1 == len(self.vertices) // 3 == len(self.texture_coords) // 2 == len(
            self.normals) // 3 == len(self.vertex_joints) // 3 == len(self.weights) // 3

        self.root_joint: Joint = root_joint
        self.key_frames = key_frames

        print(self.root_joint)
        return

    def __parse_model_data_no_texture(self):
        '''
        Currently, we don't parse the texture coordinates here for simplicity. The reason is that texture coordinates have their own
        indices which usually don't match to the vertex indices, but opengl's index buffer requires them to match. In short this means that
        for each unique combination of a vertex and a texture coordinate we need to create an index for it, which requires some work (see
        __parse_model_data).
        Basically the purpose of the whole indices idea and the index buffer is to save up memory, so maybe we can just duplicate the
        vertices and have one index for each vertex... but it might slow down everything.
        '''
        triset = self.model_data.geometries[0].primitives[0]

        # mesh
        indices = np.arange(len(triset.vertex_index)).flatten('C')
        vertices = triset.vertex[triset.vertex_index].flatten('C')

        texture_coords = triset.texcoordset[0][triset.texcoord_indexset[0]].flatten('C')

        # skeleton
        vertex_joints, weights = self.__parse_vertex_joints_weights()
        root_joint: Joint = self.__parse_joints()

        # animation
        key_frames = self.__parse_animation()

        vertex_joints = vertex_joints.reshape(-1, 3)[triset.vertex_index].flatten('C')
        weights = weights.reshape(-1, 3)[triset.vertex_index].flatten('C')

        return indices, vertices, texture_coords, vertex_joints, weights, root_joint, key_frames

    def __parse_vertex_joints_weights(self):
        weights = self.model_data.controllers[0].weights
        index = self.model_data.controllers[0].index

        index = [np.vstack((a[:, 0], weights[a[:, 1]].flatten())).T for a in index]         # replace weight indices with weights
        index = [a[a[:, 1].argsort()][::-1] for a in index]         # sort by weights, descending order
        index = [np.pad(a, ((0, 3), (0, 0))) for a in index]        # append 3 zero rows (can append 2, but just in case)
        index = [a[:3, :] for a in index]                           # keep only 3 joints
        index = [np.vstack((a[:, 0],  a[:, 1]/np.sum(a, axis=0)[1])).T for a in index]      # normalize sum to 1.0
        index = np.array(index)

        vertex_joints = index[:, :, 0].flatten('C')
        weights = index[:, :, 1].flatten('C')
        return vertex_joints, weights

    def __parse_joints(self) -> Joint:
        joints = []
        joints_inv_bind_matrix = self.model_data.controllers[0].joint_matrices
        # TODO not sure if the dictionary is always ordered
        # we rely on this order, because the vertex weights are indexed by this order of the joints
        for index, name in enumerate(joints_inv_bind_matrix):
            joint = Joint(index, name, np.eye(4), joints_inv_bind_matrix[name])
            joints.append(joint)

        # TODO currently we can only parse the farm boy model... it seems there is no convention for which node has the hierarchy information
        root_node = self.model_data.scenes[0].nodes[1].children[0]
        root_joint: Joint = build_joint_tree(root_node, joints)
        assert len(root_joint) == len(joints)
        root_joint._test_inverse_bind_transform()
        return root_joint

    def __parse_animation(self):
        animations = self.model_data.animations

        timestamps = animations[0].sourceById['Armature_Torso_pose_matrix-input'][:, 0]

        all_transforms = np.zeros((len(animations), len(timestamps), 16))
        for i, joint_animation in enumerate(animations):
            transforms_data = [value for key, value in joint_animation.sourceById.items() if 'output' in key][0]
            joint_transforms = transforms_data[:, 0].reshape(-1, 16)
            assert len(joint_transforms) == len(timestamps)
            all_transforms[i] = joint_transforms

        key_frames = []
        for i, timestamp in enumerate(timestamps):
            # get transforms of all joints in timestamp[i]
            current_transforms = all_transforms[:, i]

            transforms = []
            for i in range(len(current_transforms)):
                # get transform of all joints in current key frame
                joint_current_transform = current_transforms[i].reshape(4, 4)
                transforms.append(joint_current_transform)

            kf = KeyFrame(timestamp, transforms)
            key_frames.append(kf)

        return key_frames

    def __parse_model_data(self):
        all_vertices = np.array([], dtype='float32')
        all_texture_coords = np.array([], dtype='float32')
        all_indices = np.array([], dtype='int32')
        for geom in self.model_data.geometries:
            # TODO make sure what to do if there is more than one primitive
            triset = geom.primitives[0]

            if len(triset.texcoord_indexset) > 0:
                vertex_indices = triset.vertex_index.flatten('C')
                texture_indices = triset.texcoord_indexset[0].flatten('C')
                combined_indices = np.vstack((vertex_indices, texture_indices)).T
                combined_indices_unique = np.unique(combined_indices, axis=0)

                vertices = triset.vertex[combined_indices_unique[:, 0]].flatten('C')
                texture_coords = triset.texcoordset[0][combined_indices_unique[:, 1]].flatten('C')

                combined_indices = Nx2_to_array_of_tuples(combined_indices)
                combined_indices_unique = Nx2_to_array_of_tuples(combined_indices_unique)
                indices = map_arrays(combined_indices_unique, combined_indices).astype('int32')

                assert np.array_equal(vertices.reshape(-1, 3)[indices], triset.vertex[triset.vertex_index.flatten('C')])
                assert np.array_equal(texture_coords.reshape(-1, 2)[indices],
                                      triset.texcoordset[0][triset.texcoord_indexset[0].flatten('C')])
                assert vertices.size // 3 == texture_coords.size // 2
            else:
                vertices = triset.vertex.flatten('C')
                indices = triset.vertex_index.flatten('C')
                # texture_coords = np.array([])
                # TODO totally not sure about what to do if we don't have texture coordinates.
                texture_coords = np.zeros(len(vertices))

            indices += all_vertices.size // 3
            all_vertices = np.hstack((all_vertices, vertices))
            all_texture_coords = np.hstack((all_texture_coords, texture_coords))
            all_indices = np.hstack((all_indices, indices))

        return all_vertices, all_indices, all_texture_coords


def build_joint_tree(root_node, joints):
    root_joint: Joint = find_joint_by_name(joints, root_node.id)
    root_joint.local_bind_transform = root_node.matrix
    for node in root_node.children:
        if hasattr(node, 'id'):
            j = build_joint_tree(node, joints)
            root_joint.add_child(j)
    return root_joint


def find_joint_by_name(joints, name: str):
    return [j for j in joints if j.name == name][0]


def load_obj_mesh(model_path):
    mesh = trimesh.load(model_path)
    vertices = np.array(mesh.vertices.flatten('C'))
    indices = np.array(mesh.faces.flatten('C'))
    texture_coords = np.array([])
    return vertices, indices, texture_coords


def Nx2_to_array_of_tuples(array):
    tuples = [(i[0], i[1]) for i in array]
    array = np.empty(len(tuples), dtype=tuple)
    array[:] = tuples
    return array


def map_arrays(dst, origin):
    ''' map elements in origin to elements in dst.
        for each element in origin, find the index of the element in dst that equals it.
        this function assumes that such an element in dst exists.
    '''
    dst_sorted = np.argsort(dst)
    origin_pos = np.searchsorted(dst[dst_sorted], origin)
    indices = dst_sorted[origin_pos]
    return indices
