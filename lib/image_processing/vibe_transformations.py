import numpy as np
from numpy import ndarray


def canonic_projection(u: float, v: float, s: float, f=5000.0, h=224.0):
    """takes a 3d homogenus point and projects it to cannonical ([-1,1]^2) projection space.
    returns a X.
    returns a K(intrinsic camera matrix) and Rt(extrensic camera matrix)"""
    epsilon = 1e-9  # for numerical stability
    tz = (2 * f) / (h * s + epsilon)

    Rt = np.array([
        [1, 0, 0, u],
        [0, 1, 0, v],
        [0, 0, 1, tz]
    ])

    fx = fy = (2 * f) / h
    K = np.array([
        [fx, 0, 0],
        [0, fy, 0],
        [0, 0, 1]
    ])
    return K, Rt


def cannonic_to_image(h: int, w: int):
    """Transforms in projection space homogeneus 2d vector from [-1,1]^2 to image space [H,W] pixel.
    return a transformation 3x3 np array T.dot(x_hom) = pixel"""
    transformation = np.array([
        [h / 2, 0, h / 2],
        [0, w / 2, w / 2],
        [0, 0, 1]
    ], dtype=float)
    return transformation


def cannonic_crop_to_cannonic_orig(h_orig: int, w_orig: int, bbox: ndarray):
    """Transforms a cannonic homogenus vector [-1,1] in cropped image coordinates to
    original image coordinates([-1,1]).

    bbox - is the bounding box that the cropped image was cropped with.

    returns a left transformation - Tx (not xT)
    """
    x0, x1, y0, y1 = bbox

    h_crop = x1 - x0
    w_crop = y1 - y0

    sx = h_crop / h_orig
    sy = w_crop / w_orig

    center_crop_x = x0 + h_crop / 2
    center_crop_y = y0 + w_crop / 2

    center_orig_x = h_orig / 2
    center_orig_y = w_orig / 2

    tx = (center_crop_x - center_orig_x) / (h_orig / 2)
    ty = (center_crop_y - center_orig_y) / (w_orig / 2)

    transformation = np.array([
        [sx, 0, tx],
        [0, sy, ty],
        [0, 0, 1]
    ])

    return transformation


def kp3d_to_pixel_orig(h_orig: int, w_orig: int, bbox: ndarray, u: float, v: float, s: float):
    """This is the projection operator from the 3d keypoints returned by VIBE and the original image pixel coordinates."""
    K, Rt = canonic_projection(u, v, s)
    crop_to_orig = cannonic_crop_to_cannonic_orig(h_orig, w_orig, bbox)
    K = np.dot(crop_to_orig, K)
    to_pixel = cannonic_to_image(h_orig, w_orig)
    K = np.dot(to_pixel, K)
    return K, Rt


def homogenize(x: ndarray):
    """add one to the last coordinate."""
    one = np.array([1.], dtype=x.dtype)
    x_hom = np.concatenate((x, one))
    return x_hom


def dehomogenize(x_hom: ndarray):
    return x_hom[:-1] / x_hom[-1]


def swap_xy(x: ndarray):
    x_swaped = x.copy()
    x_swaped[0] = x[1]
    x_swaped[1] = x[0]
    return x_swaped


def kp_to_orig_image(h_orig: int, w_orig: int, cam_params: ndarray,
                        bbox: ndarray, kp: ndarray) -> ndarray:
    """Takes the entire keypoints in [-1,1] cropped image space
    and transforms them into 2d pixel space of the original(uncropped) image.

    excpects the kp to be of shape 49x2, or 49x3, will infer the correct transformation based on the
    dimentsions
    """
    if len(kp.shape) != 2:
        raise ValueError('kp should be with 2 axis. got: ', len(kp.shape))
    if kp.shape[1] not in [2, 3]:
        raise ValueError('kp should be 2 dimensional of 3 dimensional, got: ', kp.shape[1])

    # 2d or 3d flag
    is_3d = (kp.shape[1] == 3)
    # swap x,y
    kp[:, [1, 0]] = kp[:, [0, 1]]
    # homogenize
    ones = np.ones((kp.shape[0], 1), dtype=kp.dtype)
    kp = np.concatenate([kp, ones], axis=1)
    # transpose
    kp = kp.T
    # get the transformation from canonical cropped image to the canonical original image
    crop_to_orig = cannonic_crop_to_cannonic_orig(h_orig, w_orig, bbox)
    # from canonical space to pixel space
    to_pixel = cannonic_to_image(h_orig, w_orig)
    # calculate the overall transformation
    transformation = np.matmul(to_pixel, crop_to_orig)
    # add projection if it is 3d vector
    if is_3d:
        s, v, u = cam_params
        K, Rt = canonic_projection(u, v, s)
        transformation = np.matmul(transformation, K)
        transformation = np.matmul(transformation, Rt)
    # apply transformation
    kp = np.matmul(transformation, kp)
    # de homogenize the vectors
    kp = kp[:-1] / kp[-1]
    return kp.T


