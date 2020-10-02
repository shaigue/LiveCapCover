"""This is a visual test for the starting parameters of the optimization."""
import numpy as np

import config
from lib.utils.model import Model
from lib.data_utils.livecap_dataset_adapter import LiveCapAdapter
from lib.utils.camera import Camera
from lib.utils.renderer import Renderer, RenderMode, view_image_blocking, Color, draw_pixels_on_image

# basic rendering test:
# loading the modules
model = Model.from_file(config.model_path)
dataset = LiveCapAdapter(config.livecap_dataset_path, model, config.fx_fy_t_path)
camera = Camera(config.camera_to_world_matrix, dataset.image_h, dataset.image_w, dataset.fx, dataset.fy)
renderer = Renderer(RenderMode.image, model, camera=camera, joint_indices=dataset.joint_indices, **config.scale)
blocking_renderer = Renderer(RenderMode.blocking, model, camera=camera, joint_indices=dataset.joint_indices, **config.scale)

# initialization
initial_pose = model.get_initial_pose()
initial_pose.root_translation = dataset.get_initial_translation()
model.apply_livecap_pose(initial_pose)

# validating on the first entry
entry = dataset[0]
# draw the frame
view_image_blocking(entry.frame, 'frame')
# draw the 3d key points, and the the model's skeleton
blocking_renderer.draw_skeleton(entry.kp_3d + entry.kp_3d_translation, show_both=True)
# project the 3d points and compare the 2d points of vibe
p_3d = model.get_joints_positions()
projected = camera.project(p_3d)
image = np.zeros((camera.image_h, camera.image_w, 3), dtype=np.uint8)
draw_pixels_on_image(image, entry.kp_2d, Color.red)
draw_pixels_on_image(image, projected, Color.blue)
view_image_blocking(image, 'red=vibe, blue=projected 3d points')
image += renderer.draw_model()
view_image_blocking(image, 'on top of the rendered model')
# draw only the face vertices:
face_veritces = model.get_face_vertices()
projected = camera.project(face_veritces)
image = np.zeros_like(image)
draw_pixels_on_image(image, projected, Color.blue)
view_image_blocking(image, 'blue=projected face vertices')


# move on the x axis
pose = initial_pose
t0 = pose.root_translation.copy()
diff = np.array([2, 2, 5])
values = np.linspace(pose.root_translation - diff, pose.root_translation + diff, 4)

for v in values:
    pose.root_translation[0] = v[0]
    model.apply_livecap_pose(pose)
    image = renderer.draw_model()
    view_image_blocking(image)

pose.root_translation = t0.copy()
# move on the y axis
for v in values:
    pose.root_translation[1] = v[1]
    model.apply_livecap_pose(pose)
    image = renderer.draw_model()
    view_image_blocking(image)
# move on the z axis
pose.root_translation = t0.copy()

for v in values:
    pose.root_translation[2] = v[2]
    model.apply_livecap_pose(pose)
    image = renderer.draw_model()
    view_image_blocking(image)

pose.root_translation = t0.copy()

# rotate the root
root_index = model.get_root_joint_index()
values = np.linspace(-np.pi, np.pi, 5)
for ax in [0, 1, 2]:
    init_angle = pose.angles[root_index, ax]
    for v in values:
        pose.angles[root_index, ax] = v
        model.apply_livecap_pose(pose)
        image = renderer.draw_model()
        view_image_blocking(image)
    pose.angles[root_index, ax] = init_angle




