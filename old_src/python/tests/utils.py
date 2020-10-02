import math

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from matplotlib import cm

import bounding_box_tracking as bbt
import torch
import cv2 as cv


def numpy_img_batch_to_torch(batch: ndarray):
    """Converts an image batch from np.array image to torch tensor.
    :param batch: np.array with shape (N,H,W,C) where the colors range from 0 - 255
    :returns torch.Tensor with shape (N,C,H,W) where the colors range from 0 - 1
    """
    return torch.from_numpy(batch).permute(0, 3, 1, 2) / 255.0


def bbox(img_seq):
    device = 'cpu'
    tracker = bbt.VideoTracker(device)

    with torch.no_grad():
        # img_seq = torch.from_numpy(img_seq)
        img_seq = numpy_img_batch_to_torch(img_seq).to(device)
        track_res = tracker.track(img_seq)

        # get the last person tracked
        try:
            _, track_res = track_res.popitem()
        except KeyError:
            print('could not find any person in the video.')
            return ([], [])

        # bounding box for each frame in 'frames_idx'
        bbox = track_res['bbox']
        # the frames the person was detected in
        frames_idx = track_res['frames']
        return (frames_idx, bbox)


def drawContour(img, contour):
    radius = 1
    for i in range(len(contour)):
        (x, y) = contour[i]
        cv.circle(img, (x, y), radius, (0, 0, 255), -1)
    return img


def createDiagonalContour(img_width, img_height):
    contour = np.empty((0, 2), dtype=int)
    x_start = int(img_width/4)
    x_end = int(img_width*3/4)
    x_step = int((x_end - x_start)/10)
    y_start = int(img_height/4)
    y_end = int(img_height*3/4)
    y_step = int((y_end - y_start)/10)
    y = y_start
    for x in range(x_start, x_end, x_step):
        contour = np.vstack((contour, [x, y]))
        y += y_step
    return contour


def createCircleContour(img_width, img_height, radius=100):
    contour = np.empty((0, 2), dtype=int)
    for a in range(0, 360, 30):
        a = np.deg2rad(a)
        x = int(radius * math.cos(a) + (img_width/2))
        y = int(radius * math.sin(a) + (img_height/2))
        contour = np.vstack((contour, [x, y]))
    return contour


def getImageContours(img):
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 127, 255, 0)

    cv.namedWindow("thresh", cv.WINDOW_AUTOSIZE)
    cv.imshow("thresh", thresh)
    cv.waitKey(1)

    contours, hierarchy = cv.findContours(
        thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    return (contours, hierarchy)


def getRect(width=512, height=512):
    img = np.zeros((width, height), np.uint8)
    (x1, y1) = map(int, (width/4, width/4))
    (x2, y2) = map(int, (width*3/4, width*3/4))
    cv.rectangle(img, (x1, y1), (x2, y2), 255, cv.FILLED)
    return img


def shiftImg(img, x_shift, y_shift):
    rows = img.shape[0]
    cols = img.shape[1]
    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    dst = cv.warpAffine(img, M, (cols, rows))
    return dst


def getIDT(img):
    IDT = cv.distanceTransform(img, cv.DIST_L2, 3)
    return IDT


def imgDisplayCV(img, title):
    cv.namedWindow(title, cv.WINDOW_AUTOSIZE)
    cv.imshow(title, img)
    return


# get the mean image of imgs
def imgGetMean(imgs):
    result = np.zeros(imgs[0].shape, dtype=float)
    tmp = np.zeros(imgs[0].shape, dtype=float)
    for i in range(len(imgs)):
        tmp = np.float64(imgs[i])
        result += tmp
    result /= len(imgs)
    result = np.uint8(result)
    return result


# 2D Convolution
def imgGetFiltered(img, kernel=(5, 5)):
    kernel_size = kernel[0]*kernel[1]
    kernel = np.ones(kernel, np.float32)/kernel_size
    filtered = cv.filter2D(img, -1, kernel)
    return filtered


def img_get_box_filter(img, kernel=(5, 5)):
    filtered = cv.boxFilter(img, -1, kernel, normalize=False)
    return filtered


# This is done by convolving an image with a normalized box filter. It simply takes the average of all the pixels under
# the kernel area and replaces the central element. This is done by the function cv.blur() or cv.boxFilter().
def imgGetAveraged(img, kernel=(5, 5)):
    smoothed = cv.blur(img, kernel)
    return smoothed


# In this method, instead of a box filter, a Gaussian kernel is used. It is done with the function, cv.GaussianBlur().
def imgGetGaussianBlur(img):
    blur = cv.GaussianBlur(img, (5, 5), sigmaX=0)
    return blur


# Here, the function cv.medianBlur() takes the median of all the pixels under the kernel area and the central element
# is replaced with this median value. This is highly effective against salt-and-pepper noise in an image
def imgGetMedianBlur(img):
    median = cv.medianBlur(img, 5)
    return median


def imgGetDenoised(img):
    dst = cv.fastNlMeansDenoising(
        img, h=10, templateWindowSize=7, searchWindowSize=21)
    return dst


def plot2DArrayIn3D(arr, contour=[], grads=[]):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    rows, cols = arr.shape
    x = range(cols)
    y = range(rows)
    X, Y = np.meshgrid(x, y)

    # ax.plot_wireframe(X, Y, arr, rstride=20, cstride=20)
    surf = ax.plot_surface(X, Y, arr, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # plot gradient vectors
    xs = contour[:, 0]
    ys = contour[:, 1]
    zs = [arr[x][y] for x, y in zip(xs, ys)]
    grad_xs = [z[0] for z in grads]
    grad_ys = [z[1] for z in grads]
    grad_zs = np.zeros(len(contour))
    ax.scatter(xs, ys, zs, c='r', marker='o')
    ax.quiver(xs, ys, zs, grad_xs, grad_ys, grad_zs, color=['r'], length=3)

    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')

    plt.show(block=False)
    plt.pause(3)
    plt.close()


def getImgGradient(img):
    gradient = np.gradient(img)
    return gradient


#####################################################################################
# normal

def calc_tangent(arr, normalize=True):
    dx_dt = np.gradient(arr[:, 0])
    dy_dt = np.gradient(arr[:, 1])

    velocity = np.array([[dx_dt[i], dy_dt[i]] for i in range(dx_dt.size)])
    # plot_vectors(arr, velocity, 'Velocity')

    ds_dt = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)
    tangent = np.array([1/ds_dt] * 2).transpose() * velocity

    return tangent if normalize else velocity


# for each point calculate the tangent and then rotate it 90 degrees counterclockwise
# (this is a formal definition of normal)
def calc_normal(arr, normalize=True):
    tangent = calc_tangent(arr, normalize)
    # plot_vectors(arr, tangent, 'Tangent')

    tangent_x = tangent[:, 0]
    tangent_y = tangent[:, 1]
    dT_dt = np.array([[-tangent_y[i], tangent_x[i]]
                      for i in range(tangent_x.size)])
    return dT_dt


# this function also calculates a normal, except that here the normal is the gradient of the tangent
# (instead of just rotating it). therefore, there are two differences comparing to the regular calc_normal:
# 1. the direction isn't always correct.
# 2. when two following tangent vectors are the same, their graident is [0,0] and therefore the normal
#   is [0,0] (which is not something we want...)
def calc_normal2(arr):
    tangent = calc_tangent(arr)
    # plot_vectors(arr, tangent, 'Tangent')

    tangent_x = tangent[:, 0]
    tangent_y = tangent[:, 1]
    deriv_tangent_x = np.gradient(tangent_x)
    deriv_tangent_y = np.gradient(tangent_y)

    dT_dt = np.array([[deriv_tangent_x[i], deriv_tangent_y[i]]
                      for i in range(deriv_tangent_x.size)])
    # plot_vectors(arr, dT_dt, 'dT_dt')

    length_dT_dt = np.sqrt(
        deriv_tangent_x * deriv_tangent_x + deriv_tangent_y * deriv_tangent_y)
    normal = np.array([1/length_dT_dt] * 2).transpose() * dT_dt

    return normal

#####################################################################################
