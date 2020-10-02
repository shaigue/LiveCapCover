import OpenGL.GL as gl
import numpy as np

from Scene import Camera
from Entity import Entity
from utils import create_view_matrix


class ShaderProgram:
    def __init__(self, vertex_file='src/glsl/vertexShader.glsl', fragment_file='src/glsl/fragmentShader.glsl'):
        self.vertex_shader_ID = load_shader(vertex_file, gl.GL_VERTEX_SHADER)
        self.fragment_shader_ID = load_shader(fragment_file, gl.GL_FRAGMENT_SHADER)
        self.program_ID = gl.glCreateProgram()

        gl.glAttachShader(self.program_ID, self.vertex_shader_ID)
        gl.glAttachShader(self.program_ID, self.fragment_shader_ID)
        self.__bind_attributes()
        gl.glLinkProgram(self.program_ID)
        gl.glValidateProgram(self.program_ID)

        self.transformation_matrix_location = None
        self.projection_matrix_location = None
        self.view_matrix_location = None
        self.joint_transforms_location = None
        # self.texture_sampler_location = None
        self.__get_all_uniform_locations()

    def __get_all_uniform_locations(self):
        self.transformation_matrix_location = self.__get_uniform_location("transformation_matrix")
        self.projection_matrix_location = self.__get_uniform_location("projection_matrix")
        self.view_matrix_location = self.__get_uniform_location("view_matrix")
        self.joint_transforms_location = self.__get_uniform_location("joint_transforms")
        # self.texture_sampler_location = self.__get_uniform_location("texture_sampler")

    def __get_uniform_location(self, uniform_name: str):
        return gl.glGetUniformLocation(self.program_ID, uniform_name)

    def __bind_attributes(self):
        self.__bind_attribute(0, "position")
        self.__bind_attribute(1, "texture_coords")
        self.__bind_attribute(2, "normal")
        self.__bind_attribute(3, "joint_indices")  # vertex_joints
        self.__bind_attribute(4, "weights")  # weights
        return

    def __bind_attribute(self, attribute_number: int, variable_name: str):
        gl.glBindAttribLocation(self.program_ID, attribute_number, variable_name)

    def start(self):
        gl.glUseProgram(self.program_ID)

    def stop(self):
        gl.glUseProgram(0)

    def clean_up(self):
        self.stop()
        gl.glDetachShader(self.program_ID, self.vertex_shader_ID)
        gl.glDetachShader(self.program_ID, self.fragment_shader_ID)
        gl.glDeleteShader(self.vertex_shader_ID)
        gl.glDeleteShader(self.fragment_shader_ID)
        gl.glDeleteProgram(self.program_ID)

    def load_transformation_matrix(self, matrix: np.array):
        load_uniform_matrix(self.transformation_matrix_location, matrix)

    def load_projection_matrix(self, matrix: np.array):
        load_uniform_matrix(self.projection_matrix_location, matrix)

    def load_view_matrix(self, camera: Camera):
        view_matrix = create_view_matrix(camera)
        load_uniform_matrix(self.view_matrix_location, view_matrix)

    def load_joint_transforms_array(self, entity: Entity):
        joint_transforms = entity.animated_model.get_joints_transforms()
        # Im not sure about this for loop, because Im assuming the location is joint_transforms_location + i.
        # TODO Make sure how in general we should load an *array* of uniform matrices to the shader.
        for i in range(len(joint_transforms)):
            load_uniform_matrix(self.joint_transforms_location+i, joint_transforms[i])


def load_shader(file, shader_type) -> int:
    with open(file, 'r') as shader_source:
        shader_source = "".join([line for line in shader_source])
    shader_ID = gl.glCreateShader(shader_type)
    gl.glShaderSource(shader_ID, shader_source)
    gl.glCompileShader(shader_ID)
    if gl.glGetShaderiv(shader_ID, gl.GL_COMPILE_STATUS) != gl.GL_TRUE:
        raise RuntimeError(gl.glGetShaderInfoLog(shader_ID))
    return shader_ID


def load_uniform_float(location: int, value: float):
    gl.glUniform1f(location, value)


def load_uniform_float_vector(location: int, vec):
    gl.glUniform3f(location, vec[0], vec[1], vec[2])


def load_uniform_boolean(location: int, value: bool):
    value = 1.0 if value else 0.0
    gl.glUniform1f(location, value)


def load_uniform_matrix(location: int, matrix: np.array):
    # TODO not sure how to pass the matrix. note that matrix should be colomn major, and np.array is row major,
    # so this is why we pass GL_TRUE for transpose
    assert matrix.shape == (4, 4)
    gl.glUniformMatrix4fv(location, 1, gl.GL_TRUE, matrix)
