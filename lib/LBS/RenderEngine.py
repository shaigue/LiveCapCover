import math
import numpy as np
import OpenGL.GL as gl
from scipy.spatial.transform import Rotation
from lib.LBS.WindowManager import WindowManager
from lib.LBS.Scene import create_view_matrix
from lib.LBS.Shader import ShaderProgram
from lib.LBS.Scene import Scene


class RenderEngine:
    """ This class represents the entire render engine. """
    FPS_CAP = 100

    def __init__(self, window_width=1280, window_height=720):
        ''' Initializes a new render engine. Creates the display and inits the renderers. '''
        # has to be the first call because it calls glfw.init()
        self.wm = WindowManager(window_width, window_height, RenderEngine.FPS_CAP)
        self.renderer = Renderer()

    def update(self):
        ''' Updates the display. '''
        self.wm.update()

    def render_scene(self, scene: Scene):
        '''
        Renders the scene to the screen.
        scene - the game scene.
        '''
        self.renderer.render_scene(scene)

    def close(self):
        ''' Cleans up the renderers and closes the display. '''
        self.renderer.clean_up()
        self.wm.close_window()      # has to be the last call, because it calls glfw.terminate()

    def window_should_close(self):
        return self.wm.window_should_close()


class Renderer:
    FOV = 70
    NEAR_PLANE = 0.1
    FAR_PLANE = 1000

    def __init__(self):
        self.shader = ShaderProgram()

        # TODO use get_camera_projection_matrix instead
        self.projection_matrix = create_projection_matrix()
        self.shader.start()
        self.shader.load_projection_matrix(self.projection_matrix)
        # gl.glUniform1i(self.shader.texture_sampler_location, 0)
        self.shader.stop()
        return

    def prepare(self):
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClearColor(0, 1, 0, 1)     # green
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    def render_scene(self, scene):
        self.prepare()
        self.shader.start()
        self.render(scene, self.shader)
        self.shader.stop()

    def render(self, scene: Scene, shader: ShaderProgram):
        entity = scene.entity
        animated_model = entity.animated_model
        model = animated_model.raw_model
        gl.glBindVertexArray(model.vaoID)
        gl.glEnableVertexAttribArray(0)
        gl.glEnableVertexAttribArray(1)
        gl.glEnableVertexAttribArray(2)
        gl.glEnableVertexAttribArray(3)
        gl.glEnableVertexAttribArray(4)

        trans_matrix: np.ndarray = create_transformation_matrix(entity.position, entity.rotation, entity.scale)
        view_matrix: np.ndarray = create_view_matrix(scene.camera)
        joint_transforms: np.ndarray = entity.animated_model.get_joints_transforms()
        shader.load_transformation_matrix(trans_matrix)
        shader.load_view_matrix(view_matrix)
        shader.load_joint_transforms_array(joint_transforms)

        # TODO notice this call. we pass GL_TEXTURE0 because of uniform sampler2D in the fragment shader (tutorial 6)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, animated_model.textureID)
        # glDrawElements is the function that let's opengl know that we're using an index buffer (indices)
        gl.glDrawElements(gl.GL_TRIANGLES, model.vertex_count,
                          gl.GL_UNSIGNED_INT, gl.ctypes.c_void_p(0))    # TODO: not sure if c_void_p or something else

        gl.glDisableVertexAttribArray(0)
        gl.glDisableVertexAttribArray(1)
        gl.glDisableVertexAttribArray(2)
        gl.glDisableVertexAttribArray(3)
        gl.glDisableVertexAttribArray(4)
        gl.glBindVertexArray(0)

    def clean_up(self):
        self.shader.clean_up()


# def get_camera_projection_matrix():
#     # TODO call calibrateCamera to get the camera matrix
#     K = cv.calibrateCamera()
#     R = np.eye(3)
#     t = np.array([[0, 0, 0]])
#     projection_matrix = cv.projectionFromKRt(K, R, t)
#     return projection_matrix


def create_projection_matrix():
    WINDOW_WIDTH = 1280
    WINDOW_HEIGHT = 720
    aspect_ratio = float(WINDOW_WIDTH) / float(WINDOW_HEIGHT)
    angle = np.deg2rad(Renderer.FOV/2.0)
    # TODO according to the formula in the tutorial we shouldn't multiply by aspect_ratio, even though that's what he does in the code
    y_scale = (1.0 / math.tan(angle)) * aspect_ratio
    x_scale = y_scale / aspect_ratio
    frustum_length = Renderer.FAR_PLANE - Renderer.NEAR_PLANE

    projection_matrix = np.eye(4)
    projection_matrix[0][0] = x_scale
    projection_matrix[1][1] = y_scale
    projection_matrix[2][2] = -((Renderer.FAR_PLANE + Renderer.NEAR_PLANE) / frustum_length)
    projection_matrix[2][3] = -((2 * Renderer.FAR_PLANE * Renderer.NEAR_PLANE) / frustum_length)
    projection_matrix[3][2] = -1
    projection_matrix[3][3] = 0

    return projection_matrix


def create_transformation_matrix(translation: list, rotation: list, scale: float) -> np.ndarray:
    result: np.ndarray = np.eye(4)
    rotation = Rotation.from_euler('xyz', rotation, degrees=True).as_matrix()
    result[:3, :3] = rotation
    result[:3, 3] = translation
    # TODO not sure about the scale. should we just multiply it by the whole matrix?
    return scale*result
