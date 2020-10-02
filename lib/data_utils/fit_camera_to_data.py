"""This script is for optimizing the camera parameters for vibe based on the 3d / 2d correspondence.

We assume that:
* rotation is the identity function.
* cx, cy is exactly the image center
* camera has no distortions

We optimize in 2 stages:

stage 1: find optimal fx, fy.
    - we take a random sample of N frames, out of the entire dataset
    - and we solve the non linear least squares for {(tx_i, ty_i, tz_i)} i<=N, and {fx, fy}
    - we return the optimal fx, fy
    TODO: Check that the resulting fx, fy are consistent and do not depend a lot on the samples taken
stage 2: for each frame, find the optimal tx, ty, tz
    - using the estimated fx,fy in the last step we optimize tx,ty,tz for each frame with a linear least square solution
"""
import numpy as np
from numpy import ndarray
from numpy import linalg
from scipy import optimize


import config
from lib.data_utils.livecap_dataset import LiveCapDataset
from lib.utils.camera import get_no_rotation_projection_matrix, project_points

N_SAMPLES_FOR_ESTIMATING_F = 64
DATASET_PATH = config.original_dataset_path
CAMERA_PARAMS_PATH = config.original_camera_path
INTRINSIC_PARAMS_PATH = config.intrinsic_params_path
# INTRINSIC_PARAMS_PATH = None


def focal_length_residuals(opt: ndarray, cx: float, cy: float, p_2d: ndarray, p_3d: ndarray):
    fx, fy = opt[:2]
    t = opt[2:].reshape((-1, 3))

    p_projected = np.empty_like(p_2d)
    n_frames = p_2d.shape[0]
    for i in range(n_frames):
        projection_matrix = get_no_rotation_projection_matrix(fx, fy, *t[i], cx, cy)
        p_projected[i] = project_points(p_3d[i], projection_matrix)

    return (p_projected - p_2d).reshape(-1)


def estimate_focal_lengths(dataset: LiveCapDataset, n_samples: int, cx: float, cy: float):
    """For estimating the focal lengths, fx, fy.
    N - the number of random samples
    M - the number of points for each sample.
    """
    # choose N random data points
    all_indices = np.arange(len(dataset))
    sample_indices = np.random.choice(all_indices, n_samples, replace=False)
    p_2d = []
    p_3d = []
    for s in sample_indices:
        sample = dataset[s]
        p_2d.append(sample.vibe.kp_2d)
        p_3d.append(sample.vibe.kp_3d)
    # now they are in the shape (N, M, 3) and (N, M, 2) respectively
    p_2d = np.stack(p_2d)
    p_3d = np.stack(p_3d)

    n_points_per_sample = p_2d.shape[1]
    # the optimization vector is organized in the fallowing way:
    # (fx, fy, tx_0, ty_0, tz_0, ..., tx_N-1, ty_N-1, tz_N-1)
    # initialized to (1, 1) and all the t's to standard normal distribution (0,1)
    # t0 = np.random.standard_normal(n_samples * n_points_per_sample * 3)
    # f0 = np.array([1, 1])
    tx = np.full(n_samples, 0)
    ty = np.full(n_samples, 1)
    tz = np.full(n_samples, 5)
    t0 = np.stack((tx, ty, tz), axis=-1).reshape(-1)
    f0 = np.array([cx, cy])
    opt0 = np.concatenate((f0, t0))
    lower_bound = np.concatenate(([100, 100], np.full(n_samples * 3, -10)))
    upper_bound = np.concatenate(([5000, 5000], np.full(n_samples * 3, 10)))
    opt_result = optimize.least_squares(focal_length_residuals, opt0, bounds=(lower_bound, upper_bound),
                                        args=(cx, cy, p_2d, p_3d))

    return opt_result.x[:2]


def estimate_translations(dataset: LiveCapDataset, fx: float, fy: float, cx: float, cy: float):
    """Estimates the translation for each of the frames, based on the calculated fx, fy.
    for each frame, take the 3d keypoints and the 2d keypoints and and solve a linear least square problem.

    we find tx, ty, tz using least square method, by fixing fx, fy as a constant
        those least square conditions were set with simple perspective projection constraints,
                                                                    -------A_x-------    ----b_x----
        u = (f / (z + tz)) * (x + tx) -> u*z + u*tz = f*x + f*tx -> f*tx + 0*ty - u*tz - (u*z - f*x) = 0
                                                                    -------A_y-------    ----b_y----
        v = (f / (z + tz)) * (y + ty) -> v*z + v*tz = f*y + f*ty -> 0*tx + f*ty - v*tz - (v*z - f*y) = 0
        find t = (tx, ty, tz) s.t. ||At - b|| ^ 2 is minimal
    """
    n_frames = len(dataset)
    t = np.empty((n_frames, 3))
    for i in range(n_frames):
        entry = dataset[i]
        p_3d = entry.vibe.kp_3d
        p_2d = entry.vibe.kp_2d
        n_points = p_2d.shape[0]
        # separate the columns of the data points for readability
        x = p_3d[:, 0]
        y = p_3d[:, 1]
        z = p_3d[:, 2]
        # u = p_2d[:, 0]
        # v = p_2d[:, 1]
        v = p_2d[:, 0]
        u = p_2d[:, 1]

        ax = np.zeros((n_points, 3))
        ax[:, 0] = fx
        ax[:, 2] = cx - u
        bx = (u * z) - (cx * z) - (fx * x)

        ay = np.zeros((n_points, 3))
        ay[:, 1] = fy
        ay[:, 2] = cy - v
        by = (v * z) - (cy * z) - (fy * y)

        a = np.concatenate((ax, ay), axis=0)
        b = np.concatenate((bx, by), axis=0)

        opt_result = linalg.lstsq(a, b, rcond=None)
        t[i] = opt_result[0]
    return t


if __name__ == "__main__":
    dataset = LiveCapDataset(DATASET_PATH)
    if INTRINSIC_PARAMS_PATH is None:
        # fix cx and cy to be the center of the image
        cx = dataset.image_width / 2
        cy = dataset.image_height / 2
        fx, fy = estimate_focal_lengths(dataset, N_SAMPLES_FOR_ESTIMATING_F, cx, cy)
        t = estimate_translations(dataset, fx, fy, cx, cy)
        print(f'finished, got t: {t.shape}, fx={fx}, fy={fy}')
        np.savez(CAMERA_PARAMS_PATH, fx=fx, fy=fy, t=t)
    else:
        intrinsic = np.load(INTRINSIC_PARAMS_PATH)
        t = estimate_translations(dataset, intrinsic['fx'], intrinsic['fy'], intrinsic['u'], intrinsic['v'])
        np.savez(CAMERA_PARAMS_PATH, fx=intrinsic['fx'], fy=intrinsic['fy'], t=t)




