"""This is a script to validate that the model is animating correctly."""

import config
from lib.IO.read_collada import read_collada_file
from lib.utils.model import Model, LivecapPose
from lib.utils.renderer import Renderer, RenderMode
from lib.utils.camera import Camera
from lib.utils.utils import load_object

from pathlib import Path
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename


def run_animation():
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    path = Path(askopenfilename())  # show an "Open" dialog box and return the path to the selected file
    print("path: " + str(path))

    # path = config.animations_path / 'experiment'
    optimization_result = load_object(path)
    animation = optimization_result["animation"]
    model = optimization_result["model"]
    renderer = Renderer(RenderMode.video, model, image_w=config.image_width, image_h=config.image_height,
                        filename=path.with_suffix('.mp4'))
    vtk_camera = Camera.matrix_to_vtk_camera_pose(config.camera_to_world_matrix)
    renderer.set_vtk_camera_pose(vtk_camera)

    for i, livecap_pose in enumerate(animation):
        print(f'writing frame #{i}...')
        model.apply_livecap_pose(livecap_pose)
        renderer.draw_model()


def draw_blender_animation():
    raw_model_data = read_collada_file(config.model_path)
    model = Model.from_raw_model_data(raw_model_data)
    animations = raw_model_data.get_animation()
    renderer = Renderer(RenderMode.blocking, model, filename='model_animation.mp4')

    # draw the initial model
    renderer.draw_model()
    # sanity check that those functions work correctly
    initial_pose = model.get_initial_pose()
    model.apply_livecap_pose(initial_pose)
    renderer.draw_model()

    i = 0
    for keyframe in animations.key_frames:
        print(f'writing {i} frame...')
        i += 1
        # model.apply_transformation_list(keyframe.pose)
        model.apply_livecap_pose(LivecapPose.from_transformation_list(keyframe.pose))
        renderer.draw_model()

    renderer.close()


if __name__ == '__main__':
    run_animation()
