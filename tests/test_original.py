import pyvista as pv
import numpy as np

import config
from lib.original.reader import *
from lib.utils.renderer import Renderer, view_image_blocking, draw_pixels_on_image, DebugRenderer
from lib.utils.model import LivecapModel, Model
from lib.utils.camera import Camera, project_points
from lib.data_utils.livecap_dataset_adapter import LiveCapAdapter, get_vibe_to_original_joints, \
    get_vibe_model_dimensions
from lib.skeletal_pose_estimation.optimization2 import load_optimization_settings


def test_read_motion():
    vectors = read_motion(config.motion_path)
    print(vectors.dtype)
    print(vectors.shape)
    print(vectors[0])


def test_read_calibration():
    calibration_list = read_calibration(config.calibration_path)
    print(len(calibration_list))
    print(calibration_list[0])


def test_read_obj():
    obj = read_obj(config.obj_path)
    faces = obj['faces']
    threes = np.full((faces.shape[0], 1), faces.shape[1])
    faces = np.concatenate((threes, faces), axis=1)
    mesh = pv.PolyData(obj['vertices'], faces)
    mesh.t_coords = obj['vertex_texture_coords']
    texture = pv.read_texture(config.texture_path)
    mesh.plot(texture=texture)


def test_read_segmentation():
    seg = read_segmentation(config.segmentation_path)
    print(seg.dtype, seg.shape)


def test_read_skin():
    bones, weights = read_skin(config.skin_path)
    print(bones)
    print(weights.dtype, weights.shape)
    print(weights[:3])


def test_read_skeleton():
    read_skeleton(config.skeleton_path)


def test_read_skinning_data():
    res = read_skinning_data(config.skin_path, config.skeleton_path)
    print(res)


def test_read_livecap_model():
    raw = read_livecap_model(*config.read_livecap_model_args)
    print(raw)


def test_model_rendering():
    raw = read_livecap_model(*config.read_livecap_model_args)
    model = LivecapModel.from_raw_model_data(raw)
    renderer = Renderer('blocking', model, show_axes=True)
    renderer.draw_model(with_texture=True)
    renderer.draw_skeleton()
    motion = read_motion(config.motion_path)
    # each row is a motion plan
    for i in range(0, len(motion), 100):
        vector = motion[i]
        vector[:3] *= model.scale
        model.apply_pose_vector(vector)
        renderer.draw_skeleton()
        renderer.draw_model(with_texture=True)


def test_model_rendering2():
    a = load_optimization_settings(use_scale=True, mode='blocking')
    model, dataset, camera, renderer, initial_pose_vector = a
    model.apply_pose_vector(initial_pose_vector)
    renderer.draw_model(with_texture=True)
    # renderer.draw_skeleton()
    motion = read_motion(config.motion_path)
    # each row is a motion plan
    t = Transformation(config.t_livecap_to_camera)
    for i in range(len(motion)):
        vector = motion[i]
        vector[:3] *= model.scale
        model.apply_pose_vector(vector)
        # renderer.draw_skeleton()
        renderer.draw_model(with_texture=True)


def test_adapter_match():
    raw = read_livecap_model(*config.read_livecap_model_args)
    model = LivecapModel.from_raw_model_data(raw)
    dataset = LiveCapAdapter(config.original_dataset_path, model, config.original_camera_path,
                             get_vibe_to_original_joints())

    cam_params = np.load(config.intrinsic_params_path)
    fx = cam_params['fx']
    fy = cam_params['fy']
    u = cam_params['u']
    v = cam_params['v']
    h = cam_params['h']
    w = cam_params['w']

    renderer = DebugRenderer(w, h, model, dataset.joint_indices)

    # project points with the original with the dataset parameters
    camera = Camera(config.camera_to_world_matrix, h, w, fx, fy, u, v)
    datapoint = dataset[0]
    kp3d = datapoint.kp_3d
    kp3d_translation = datapoint.kp_3d_translation
    p3d = model.get_p3d()[dataset.joint_indices] + kp3d_translation
    renderer.debug_3d(p3d, kp3d, kp3d_translation)
    kp3d = kp3d + kp3d_translation
    kp2d = camera.project(kp3d)
    kp2d_true = datapoint.kp_2d
    p2d = camera.project(p3d)
    frame = datapoint.frame
    draw_pixels_on_image(frame, p2d, 'red')
    draw_pixels_on_image(frame, kp2d, 'blue')
    draw_pixels_on_image(frame, kp2d_true, 'green')
    view_image_blocking(frame)


def main():
    # test_read_motion()
    # test_read_calibration()
    # test_read_obj()
    # test_read_segmentation()
    # test_read_skin()
    # test_read_skeleton()
    # test_read_skinning_data()
    # test_read_livecap_model()
    # test_model_rendering()
    # test_adapter_match()
    test_model_rendering2()


if __name__ == "__main__":
    main()
