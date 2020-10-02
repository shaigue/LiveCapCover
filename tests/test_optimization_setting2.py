"""This is a visual test for the starting parameters of the optimization, for livecap original dataset."""
from lib.skeletal_pose_estimation.optimization2 import *
from lib.utils.renderer import view_image_blocking, draw_pixels_on_image


# basic rendering test:
# loading the modules
model, dataset, camera, renderer, initial_pose_vector = load_optimization_settings()
blocking_renderer = load_renderer(model, camera, dataset, use_scale=True, mode='blocking', filename=None)

# initialization
model.apply_pose_vector(initial_pose_vector)

# validating on the first entry
entry = dataset[0]
# draw the frame
view_image_blocking(entry.frame, 'frame')
# draw the 3d key points, and the the model's skeleton
blocking_renderer.draw_skeleton(entry.kp_3d + entry.kp_3d_translation, show_both=True)
# project the 3d points and compare the 2d points of vibe
p_3d = model.get_p3d()
projected = camera.project(p_3d)
image = np.zeros((camera.image_h, camera.image_w, 3), dtype=np.uint8)
draw_pixels_on_image(image, entry.kp_2d, 'red')
draw_pixels_on_image(image, projected, 'blue')
view_image_blocking(image, 'red=vibe, blue=projected 3d points')
image += renderer.draw_model()
view_image_blocking(image, 'on top of the rendered model')

for i in range(10):
    initial_pose_vector[2] = 3 + 4 * (i + 1) / 10
    model.apply_pose_vector(initial_pose_vector)
    image = renderer.draw_model()
    view_image_blocking(image)
