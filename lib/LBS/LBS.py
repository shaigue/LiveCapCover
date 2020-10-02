import glfw
from lib.LBS.RenderEngine import RenderEngine
from lib.LBS.Scene import Scene, Camera, Entity
from lib.LBS.OpenGLLoader import OpenGLLoader
from lib.LBS.Model import RawModel, AnimatedModel, Model
from lib.LBS.Animation import Animation
from lib.utils.raw_model_data import SkeletonPlotter
import config
import copy

import OpenGL.GL as gl
import cv2 as cv
import numpy as np
from PIL import Image
from PIL import ImageOps


def render(model: Model, animation: Animation = None):
    ''' Initialises the engine and loads the scene. For every frame it updates the
        camera, updates the animated entity (which updates the animation),
        renders the scene to the screen, and then updates the display. When the
        display is closed the engine gets cleaned up.
    '''

    engine = RenderEngine()
    loader = OpenGLLoader()

    raw_model: RawModel = loader.load_to_VAO(model)
    textureID = loader.load_texture(config.farm_boy_texture_path)

    animated_model = AnimatedModel(raw_model, textureID, root_joint=model.root_joint)
    # entity: Entity = Entity(animated_model, position=[0, -5, -30], rotation=[-90, 30, 0], scale=1.0)     # farm boy
    # entity: Entity = Entity(animated_model, position=[0, -10, -30], rotation=[0, 0, 0], scale=1.0)
    entity: Entity = Entity(animated_model, position=[0, -1, -5], rotation=[-90, 0, 0], scale=1.0)

    camera = Camera(model.vertices.ptp())
    scene = Scene(entity, camera)

    scene.entity.animator.set_animation(animation)

    plotter = SkeletonPlotter(model.root_joint)

    while not engine.window_should_close():
        glfw.poll_events()
        scene.camera.move()

        scene.entity.increase_position([0, 0, 0])
        # scene.entity.increase_rotation([0, 0.2, 0])
        scene.entity.update()
        plotter.plot()

        engine.render_scene(scene)
        engine.update()

    loader.clean_up()
    engine.close()


class ModelRenderer:
    def __init__(self, model: Model, window_width, window_height):
        self.window_width = window_width
        self.window_height = window_height
        self.root_joint = copy.deepcopy(model.root_joint)
        self.engine = RenderEngine()
        self.loader = OpenGLLoader()

        raw_model: RawModel = self.loader.load_to_VAO(model)
        textureID = self.loader.load_texture(config.farm_boy_texture_path)
        animated_model = AnimatedModel(raw_model, textureID, root_joint=self.root_joint)
        # entity: Entity = Entity(animated_model, position=[0, -1, -5], rotation=[-90, 0, 0], scale=1.0)
        entity: Entity = Entity(animated_model, position=[0, -1, -4], rotation=[-90, 0, 0], scale=1.0)
        camera = Camera(model.vertices.ptp())

        self.scene = Scene(entity, camera)
        self.engine.render_scene(self.scene)
        self.engine.update()

    def __enter__(self):
        return self

    def pose_to_img(self, pose):
        self.root_joint.apply_pose_to_joints(pose)
        self.engine.render_scene(self.scene)
        self.engine.update()

        data = gl.glReadPixels(0, 0, self.window_width, self.window_height, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
        image = Image.frombytes("RGBA", (self.window_width, self.window_height), data)
        image = ImageOps.flip(image)
        image = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
        return image

    def __exit__(self, exc_type, exc_value, traceback):
        self.loader.clean_up()
        self.engine.close()
