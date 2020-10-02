from ColladaParser import ColladaParser
from Model import RawModel

import OpenGL.GL as gl
import numpy as np


class OpenGLLoader:
    def __init__(self):
        self.vaos = []
        self.vbos = []
        self.textures = []

    def __create_VAO(self) -> int:
        vaoID = gl.glGenVertexArrays(1)
        self.vaos.append(vaoID)
        gl.glBindVertexArray(vaoID)
        return vaoID

    def __create_VBO(self) -> int:
        vboID = gl.glGenBuffers(1)
        self.vbos.append(vboID)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vboID)

    def __unbind_VAO(self):
        '''
        unbind current bound VAO.
        '''
        gl.glBindVertexArray(0)
        return

    def __unbind_VBO(self):
        '''
        unbind current bound VBO.
        '''
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    def __bind_index_buffer(self, indices: np.array):
        vboID = gl.glGenBuffers(1)
        self.vbos.append(vboID)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, vboID)

        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)

        # TODO: in the tutorial (#tutorial < 6) he didn't unbind, so Im not sure if it's needed.
        # for some reason, looking at ThinMatrix's java code we don't unbind the index buffer.
        # gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
        return

    def __store_data_in_attribute_list(self, attribute_number, coordinate_size, data: np.array):
        self.__create_VBO()
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_STATIC_DRAW)
        gl.glVertexAttribPointer(
            attribute_number, size=coordinate_size, type=gl.GL_FLOAT, normalized=gl.GL_FALSE, stride=0, pointer=gl.ctypes.c_void_p(0))
        self.__unbind_VBO()

    def __store_data_in_int_attribute_list(self, attribute_number, coordinate_size, data: np.array):
        self.__create_VBO()
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_STATIC_DRAW)
        gl.glVertexAttribIPointer(attribute_number, coordinate_size, gl.GL_INT, 0, gl.ctypes.c_void_p(0))
        self.__unbind_VBO()

    def load_to_VAO(self, model: ColladaParser) -> RawModel:
        vaoID = self.__create_VAO()
        self.__bind_index_buffer(model.indices)
        self.__store_data_in_attribute_list(attribute_number=0, coordinate_size=3, data=model.vertices)
        self.__store_data_in_attribute_list(attribute_number=1, coordinate_size=2, data=model.texture_coords)
        self.__store_data_in_attribute_list(attribute_number=2, coordinate_size=3, data=model.normals)
        self.__store_data_in_int_attribute_list(attribute_number=3, coordinate_size=3, data=model.vertex_joints)
        self.__store_data_in_attribute_list(attribute_number=4, coordinate_size=3, data=model.weights)
        self.__unbind_VAO()
        return RawModel(vaoID, len(model.indices))

    def load_texture(self, texture_path: str):
        textureID = gl.glGenTextures(1)
        self.textures.append(textureID)
        gl.glBindTexture(gl.GL_TEXTURE_2D, textureID)
        # set texture wrapping parameters
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        # set texture filtering parameters
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

        img_data, height, width, mode = img_to_bytes(texture_path)

        mode = gl.GL_RGB if mode == 'RGB' else gl.GL_RGBA
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, mode, width,
                        height, 0, mode, gl.GL_UNSIGNED_BYTE, img_data)
        return textureID

    def clean_up(self):
        gl.glDeleteBuffers(len(self.vbos), self.vbos)
        gl.glDeleteVertexArrays(len(self.vaos), self.vaos)
        gl.glDeleteTextures(self.textures)      # TODO: not sure if this is the way we should call this glDeleteTextures function


def img_to_bytes(texture_path):
    # TODO: Im not completely peaceful with this implementation. try using opencv for coherence
    from PIL import Image
    img = Image.open(texture_path)
    x = 1024
    img = img.resize((x, x))
    img_data = np.array(img.getdata(), np.uint8)

    height, width = (img.size[0], img.size[1])
    mode = img.mode
    if mode != 'RGB' and mode != 'RGBA':
        raise Exception('what the hell man why is the image neither RGB nor RGBA???')
    return img_data, height, width, mode
