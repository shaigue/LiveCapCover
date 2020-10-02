"""Module for running the livecap optimization."""
from pathlib import Path
from datetime import date
import shutil

import numpy as np
from numpy import ndarray
from scipy.optimize import least_squares


import config
from lib.data_utils.livecap_dataset_adapter import LiveCapAdapter, AdapterEntry,\
    get_vibe_to_original_joints
from lib.image_processing.image_distance_transform import get_silhouette_idt_grad_from_mask
from lib.original.reader import read_livecap_model
from lib.utils.model import LivecapModel
from lib.utils.camera import Camera
from lib.utils.renderer import Renderer, DebugRenderer, draw_pixels_on_image, view_image_blocking

# ==================================== setup ==================================


def load_livecap_model():
    raw = read_livecap_model(*config.read_livecap_model_args)
    return LivecapModel.from_raw_model_data(raw)


def load_livecap_dataset(model: LivecapModel):
    return LiveCapAdapter(
        root=config.original_dataset_path,
        model=model,
        camera_fitting_path=config.original_camera_path,
        kp_to_joint_map=get_vibe_to_original_joints(),
    )


def load_camera():
    params = np.load(config.intrinsic_params_path)
    return Camera(
        t_camera_to_world=config.camera_to_world_matrix,
        image_h=params['h'],
        image_w=params['w'],
        fx=params['fx'],
        fy=params['fy'],
        u=params['u'],
        v=params['v'],
    )


def load_renderer(model, camera, dataset, use_scale, mode, filename):
    scale = {
        'xscale': 1,
        'yscale': 1,
        'zscale': 1,
    }
    if use_scale:
        scale = np.load(config.original_scale_path)

    return Renderer(
        mode=mode,
        model=model,
        image_h=dataset.image_h,
        image_w=dataset.image_w,
        filename=filename,
        joint_indices=dataset.joint_indices,
        camera=camera,
        **scale,
    )


def get_initial_pose_vector(model, dataset):
    t = dataset.get_initial_translation()
    padding = np.zeros((model.n_dof - 3))
    return np.concatenate((t, padding))


def load_optimization_settings(use_scale=True, mode='image', filename=None, renderer=True):
    model = load_livecap_model()
    dataset = load_livecap_dataset(model)
    camera = load_camera()
    initial_pose_vector = get_initial_pose_vector(model, dataset)
    if renderer:
        renderer = load_renderer(model, camera, dataset, use_scale, mode=mode, filename=filename)
        return model, dataset, camera, renderer, initial_pose_vector
    return model, dataset, camera, initial_pose_vector


def fit_scale():
    model, dataset, camera, renderer, initial_pose_vector = load_optimization_settings(use_scale=False, renderer=True)
    # to make all the model appear in the image
    initial_pose_vector[0] += 0.4
    initial_pose_vector[2] += 1
    model.apply_pose_vector(initial_pose_vector)

    image = renderer.draw_model()
    vertices_proj = camera.project(model.vertices)

    cond = (image == renderer.model_color).all(axis=2)
    model_pixels = np.argwhere(cond)
    d_renderer = model_pixels.max(axis=0) - model_pixels.min(axis=0)
    d_proj = vertices_proj.max(axis=0) - vertices_proj.min(axis=0)
    scale = d_proj / d_renderer

    return {
        'xscale': scale[1],
        'yscale': scale[0],
        'zscale': 1
    }


def write_scale_to_file():
    scale = fit_scale()
    np.savez(config.original_scale_path, **scale)


def find_contour_vertices(vertices: ndarray, camera: Camera, n: int, axis_order='xy'):
    if n % 2 != 0:
        m = n + 1
    else:
        m = n
    # project all of the vertices
    vertices_proj = camera.project(vertices)
    # take the min-max values of the x coordinate and split it into m / 2 bins
    if axis_order == 'xy':
        x, y = vertices_proj.T
    elif axis_order == 'yx':
        y, x = vertices_proj.T
    else:
        raise RuntimeError
    min_x = np.min(x)
    max_x = np.max(x)
    n_bins = m // 2
    bin_size = (max_x - min_x) / n_bins
    contour_indices = []
    start_x = min_x
    for _ in range(n_bins):
        end_x = start_x + bin_size
        # find the indices of the points that are in this bins
        indices = np.argwhere((x > start_x) & (x <= end_x)).squeeze()
        bin_y = y[indices]
        # find the points with the maximum and minimum y value in that bin
        min_y_i = np.argmin(bin_y)
        max_y_i = np.argmax(bin_y)
        min_y_i = indices[min_y_i]
        max_y_i = indices[max_y_i]
        contour_indices.append(min_y_i)
        contour_indices.append(max_y_i)
        start_x = end_x

    return vertices_proj[contour_indices[:n]], contour_indices[:n]


def get_contour_vertices(model: LivecapModel, camera: Camera, n: int):
    p1, i1 = find_contour_vertices(model.vertices, camera, n // 2, 'xy')
    p2, i2 = find_contour_vertices(model.vertices, camera, n - n // 2, 'yx')
    pixels, indices = np.concatenate((p1, p2), axis=0), np.concatenate((i1, i2), axis=0)
    normals = model.normals[indices, :2] # remove the z element
    return pixels, normals

# ===================================== Energy ================================


class Energy:
    def __init__(self, model: LivecapModel, camera: Camera, joint_indices: ndarray,
                 c_2d: float = 0, c_3d: float = 0, c_anatomic: float = 0, c_silhouette: float = 0,
                 c_temporal: float = 0, debug: bool = False, n_contour_vertices=50):
        self.model = model
        self.camera = camera
        self.joint_indices = joint_indices
        self.c_2d = c_2d
        self.c_3d = c_3d
        self.c_anatomic = c_anatomic
        self.c_silhouette = c_silhouette
        self.c_temporal = c_temporal
        self.prev_p3d = None
        self.min_joint_angles, self.max_joint_angles = model.get_limits()
        self.debug = debug
        self.n_evaluations = 0
        self.n_contour_vertices = n_contour_vertices
        if debug:
            self.debug_renderer = DebugRenderer(camera.image_w, camera.image_h, model, joint_indices)

    def frame_end(self):
        """saves the current joint position."""
        self.prev_p3d = self.model.get_p3d()[self.joint_indices]
        self.n_evaluations = 0

    def show_debug(self):
        """Show debug information only in the first evaluation for that frame."""
        return self.debug and self.n_evaluations == 0

    def end_evaluation(self):
        self.n_evaluations += 1
        if self.debug and self.n_evaluations % 100 == 0:
            print(f'evaluation number {self.n_evaluations} for the current frame.')

    def energy_2d(self, p3d: ndarray, kp_2d: ndarray) -> ndarray:
        if self.c_2d == 0:
            return np.zeros_like(kp_2d)
        # project the points into the image coordinates
        p2d = self.camera.project(p3d)
        if self.show_debug():
            self.debug_renderer.debug_2d(p2d, kp_2d)
        return self.c_2d * (p2d - kp_2d)

    def energy_3d(self, p3d: ndarray, kp_3d: ndarray, kp_3d_translation: ndarray) -> ndarray:
        if self.c_3d == 0:
            return np.zeros_like(p3d)
        # return their difference with with the given translated positions
        if self.show_debug():
            self.debug_renderer.debug_3d(p3d, kp_3d, kp_3d_translation)
        return self.c_3d * (p3d - (kp_3d + kp_3d_translation))

    def energy_silhouette(self, image_fg_mask: ndarray) -> ndarray:
        if self.c_silhouette == 0:
            return np.zeros(self.n_contour_vertices)
        # get the boolean masks for the foreground segmentation
        image_fg_mask = image_fg_mask > 0
        # image_fg_mask is for checking if some point is in the model fg_mask
        image_silhouette, image_idt, image_grad = get_silhouette_idt_grad_from_mask(image_fg_mask)
        # get the contour vertices
        contour_pixels, contour_normals = get_contour_vertices(self.model, self.camera, n=self.n_contour_vertices)
        # round them so we can access them like indices
        contour_pixels = contour_pixels.round().astype(int)
        # clip them so we will not get out of bound
        rows, cols = contour_pixels.T
        rows = rows.clip(0, self.camera.image_h - 1)
        cols = cols.clip(0, self.camera.image_w - 1)
        # z = -image_grad
        z = -1 * (image_grad[rows, cols])
        z_dot_n = np.einsum('...i,...i', z, contour_normals)
        z_dot_n_negative = z_dot_n < 0

        inside = image_fg_mask[rows, cols]
        # b from the article
        b = np.where(inside & z_dot_n_negative, -1, 1)
        idt = image_idt[rows, cols] * b
        if self.show_debug():
            self.debug_renderer.debug_silhouette(contour_pixels, image_silhouette, image_idt)

        return self.c_silhouette * idt

    def energy_temporal(self, p3d: ndarray) -> ndarray:
        if self.c_temporal == 0 or self.prev_p3d is None:
            return np.zeros_like(p3d)

        return self.c_temporal * (p3d - self.prev_p3d)

    def energy_anatomic(self, pose_vector: ndarray) -> ndarray:
        if self.c_anatomic == 0:
            return np.zeros_like(pose_vector)
        # drop the translation
        result = np.where(pose_vector > self.max_joint_angles, pose_vector - self.max_joint_angles, 0)
        result = np.where(pose_vector < self.min_joint_angles, self.min_joint_angles - pose_vector, result)
        return self.c_anatomic * result

    def energy_pose(self, pose_vector: ndarray, entry: AdapterEntry) -> ndarray:
        # apply the new pose, and get the joint's positions
        self.model.apply_pose_vector(pose_vector)
        p3d = self.model.get_p3d()[self.joint_indices]
        # collect all the energy terms
        energies = (
            self.energy_2d(p3d, entry.kp_2d),
            self.energy_3d(p3d, entry.kp_3d, entry.kp_3d_translation),
            self.energy_silhouette(entry.silhouette),
            self.energy_anatomic(pose_vector),
            self.energy_temporal(p3d),
        )
        # flatten and concatenate it into a single vector
        result = np.concatenate(energies, axis=None)

        self.end_evaluation()
        return result


# ===================================== Optimization ==========================


def frame_optimization(entry: AdapterEntry, prev_pose: ndarray, energy: Energy) -> ndarray:
    """Function for a single frame in the optimization.
    :returns the optimization result for the current frame.
    """
    optimization_result = least_squares(energy.energy_pose, prev_pose, args=(entry,),
                                        method='lm', max_nfev=900, verbose=2)
    energy.frame_end()
    estimated_pose = optimization_result.x
    return estimated_pose


def video_optimization(dataset: LiveCapAdapter, energy: Energy, initial_pose: ndarray,
                       n_frames=-1, verbose=True) -> ndarray:
    """Optimization for the entire video

    :returns a list with an optimization result for each frame.
    """
    if n_frames <= 0:
        n_frames = len(dataset)
    if verbose:
        print(f'running video_optimization for {n_frames} frames...')

    pose_list = []
    pose = initial_pose

    for i in range(n_frames):
        if verbose:
            print(f'estimating pose in frame {i} out of {n_frames}...')

        entry = dataset[i]
        pose = frame_optimization(entry, pose, energy)
        pose_list.append(pose)

    if verbose:
        print('finished video optimization.')

    return np.array(pose_list)


def optimization_setup(experiment: bool, weights: dict):
    n_frames = -1
    if experiment:
        n_frames = 5
    model, dataset, camera, initial_pose = load_optimization_settings(renderer=False)
    energy = Energy(
        model=model,
        camera=camera,
        joint_indices=dataset.joint_indices,
        **weights,
        debug=experiment,
    )
    return dataset, energy, initial_pose, n_frames


def run_optimization(weights: dict, save_path: Path, experiment=False):
    args = optimization_setup(experiment=experiment, weights=weights)
    poses = video_optimization(*args)
    print(f'saving optimization results at {str(save_path)}')
    np.save(save_path, poses)


def render_results(video: bool, path: Path, save_path: Path = None):
    print('rendering results...')
    mode = 'blocking'
    if video:
        mode = 'video'
    model, dataset, camera, renderer, initial_pose = load_optimization_settings(mode=mode, renderer=True,
                                                                                filename=save_path)
    print('loaded...')
    poses = np.load(path)
    for i, pose in enumerate(poses):
        print(f'{i+1} frame...')
        model.apply_pose_vector(pose)
        renderer.draw_model(with_texture=True)

    if video:
        renderer.close()
        # view_video_blocking(config.original_video_path)


def test_find_contour_vertices():
    model, dataset, camera, renderer, initial_pose = load_optimization_settings(renderer=True)
    model.apply_pose_vector(initial_pose)

    cv, normals = get_contour_vertices(model, camera, 50)
    print(normals.shape)
    print(cv.shape)
    image = np.zeros((camera.image_h, camera.image_w, 3), dtype=np.uint8)
    draw_pixels_on_image(image, cv)
    view_image_blocking(image)


def create_videos():
    exps_paths = [
        # r'C:\Users\shaig\Documents\CS_Technion\2020_b\project GIP\repo\LiveCapCover\assets\experiments\2020-09-19_2',
        # r'C:\Users\shaig\Documents\CS_Technion\2020_b\project GIP\repo\LiveCapCover\assets\experiments\2020-09-19_3',
        # r'C:\Users\shaig\Documents\CS_Technion\2020_b\project GIP\repo\LiveCapCover\assets\experiments\2020-09-20_4',
        r'C:\Users\shaig\Documents\CS_Technion\2020_b\project GIP\repo\LiveCapCover\assets\experiments\t_smoothing',
    ]
    exps_paths = list(map(Path, exps_paths))
    for exp in exps_paths:
        print(f'rendering {exp.name}')
        video_path = exp / 'video.mp4'
        poses_path = exp / 'poses.npy'
        render_results(True, poses_path, video_path)


if __name__ == "__main__":
    # create_videos()
    # exit()
    # video_path = Path(r'C:\Users\shaig\Documents\CS_Technion\2020_b\project GIP\repo\LiveCapCover\assets\experiments\2020-09-17_1\video.mp4')
    # poses_path = Path(r'C:\Users\shaig\Documents\CS_Technion\2020_b\project GIP\repo\LiveCapCover\assets\experiments\2020-09-17_1\poses.npy')
    #
    # render_results(True, poses_path, save_path=video_path)
    # exit()

    n_experiments = len(config.experiment_weights)
    for i, weights in enumerate(config.experiment_weights):
        if i < 6:
            continue
        print(f'running experiment {i + 1} out of {n_experiments}...')
        experiment_dir = config.experiments_dir / f'{date.today()}_{i}'
        if experiment_dir.exists():
            shutil.rmtree(experiment_dir)
        experiment_dir.mkdir()
        poses_path = experiment_dir / 'poses.npy'
        # video_path = experiment_dir / 'video.mp4'
        # do optimization
        run_optimization(weights, poses_path, experiment=True)
        # save video of the results
        # render_results(True, poses_path, save_path=video_path)
