import numpy as np
import glfw
from RenderEngine import RenderEngine
from Scene import Scene, Camera
from OpenGLLoader import OpenGLLoader
from ColladaParser import ColladaParser
from Model import RawModel, AnimatedModel
from Entity import Entity
from Animation import Animation

vertices = [-0.5, 0.5, -0.5,
            -0.5, -0.5, -0.5,
            0.5, -0.5, -0.5,
            0.5, 0.5, -0.5,

            -0.5, 0.5, 0.5,
            -0.5, -0.5, 0.5,
            0.5, -0.5, 0.5,
            0.5, 0.5, 0.5,

            0.5, 0.5, -0.5,
            0.5, -0.5, -0.5,
            0.5, -0.5, 0.5,
            0.5, 0.5, 0.5,

            -0.5, 0.5, -0.5,
            -0.5, -0.5, -0.5,
            -0.5, -0.5, 0.5,
            -0.5, 0.5, 0.5,

            -0.5, 0.5, 0.5,
            -0.5, 0.5, -0.5,
            0.5, 0.5, -0.5,
            0.5, 0.5, 0.5,

            -0.5, -0.5, 0.5,
            -0.5, -0.5, -0.5,
            0.5, -0.5, -0.5,
            0.5, -0.5, 0.5]
vertices = np.array(vertices, dtype=np.float32)

texture_coords = [0, 0,
                  0, 1,
                  1, 1,
                  1, 0,
                  0, 0,
                  0, 1,
                  1, 1,
                  1, 0,
                  0, 0,
                  0, 1,
                  1, 1,
                  1, 0,
                  0, 0,
                  0, 1,
                  1, 1,
                  1, 0,
                  0, 0,
                  0, 1,
                  1, 1,
                  1, 0,
                  0, 0,
                  0, 1,
                  1, 1,
                  1, 0]
# texture_coords = []
texture_coords = np.array(texture_coords, dtype=np.float32)

indices = [
    0, 1, 3,
    3, 1, 2,
    4, 5, 7,
    7, 5, 6,
    8, 9, 11,
    11, 9, 10,
    12, 13, 15,
    15, 13, 14,
    16, 17, 19,
    19, 17, 18,
    20, 21, 23,
    23, 21, 22]
indices = np.array(indices, dtype=np.int)


def main():
    ''' Initialises the engine and loads the scene. For every frame it updates the
        camera, updates the animated entity (which updates the animation),
        renders the scene to the screen, and then updates the display. When the
        display is closed the engine gets cleaned up.
    '''
    # texture_path = 'data/models/assimp/duck_sample.jpg'
    # texture_path = 'data/shea-coulee.png'
    texture_path = 'data/models/farm_boy/diffuse.png'
    # model_path = 'data/models/duck/duck_triangles.dae'
    # model_path = 'data/models/basic/cow.obj'
    # model_path = 'data/models/regina/regina.dae'
    model_path = 'data/models/farm_boy/model.dae'

    engine = RenderEngine()
    loader = OpenGLLoader()

    model = ColladaParser(model_path)
    raw_model: RawModel = loader.load_to_VAO(model)
    textureID = loader.load_texture(texture_path)

    animated_model = AnimatedModel(raw_model, textureID, root_joint=model.root_joint, num_joints=16)
    entity = Entity(animated_model, position=[0, -5, -30], rotation=[-90, 30, 0], scale=1.0)     # farm boy
    # entity = Entity(animated_model, position=[0, -160, -600], rotation=[0, 0, 0], scale=1.0)     # regina

    camera = Camera(model.vertices.max())
    scene = Scene(entity, camera)

    animation = Animation(model.key_frames)
    scene.entity.animator.set_animation(animation)

    while not engine.window_should_close():
        glfw.poll_events()
        scene.camera.move()

        scene.entity.increase_position([0, 0, 0])
        # scene.entity.increase_rotation([0, 0.2, 0])
        scene.entity.update()

        engine.render_scene(scene)
        engine.update()

    loader.clean_up()
    engine.close()


if __name__ == '__main__':
    main()
