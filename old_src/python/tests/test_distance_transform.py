#!/usr/bin/env python

import pyrealsense2 as rs
import numpy as np
import cv2 as cv
import math

import utils as ut

SILH_PIXEL = 0
BG_PIXEL = 1

IMG_WIDTH = 640
IMG_HEIGHT = 480

full_silhouette = np.full((IMG_HEIGHT, IMG_WIDTH),
                          fill_value=SILH_PIXEL, dtype=np.uint8)
full_background = np.full((IMG_HEIGHT, IMG_WIDTH),
                          fill_value=BG_PIXEL, dtype=np.uint8)
out_of_buffer = 0


def test_img_gradient():
    img = np.full((512, 512), 255, np.uint8)

    # cv.circle(img, (256, 256), radius=3, color=0, thickness=-1)
    # cv.line(img, (0, 0), (0, 511), color=0, thickness=3)
    # draw diagonal on image
    cv.line(img, (0, 0), (511, 511), color=0, thickness=3)

    IDT = ut.getIDT(img)
    contour = ut.createCircleContour(IMG_WIDTH, IMG_HEIGHT)
    IDT_gradient = ut.getImgGradient(IDT)
    normals = ut.calc_normal(contour, normalize=False).astype(int)
    IDT = draw_contour_on_IDT(IDT, contour, normals, IDT_gradient)

    IDT = cv.convertScaleAbs(IDT)

    ut.imgDisplayCV(img, "img")
    ut.imgDisplayCV(IDT, "IDT")
    return


def draw_contour_on_IDT(IDT, contour, normals, IDT_gradient, plot3D=False):
    IDT_gradient_y, IDT_gradient_x = IDT_gradient

    # collect distance and gradient for all contour vertices
    dists = []
    grads = []
    for (x, y) in contour:
        d = IDT[y][x]
        z = np.asanyarray(
            [IDT_gradient_x[y][x], IDT_gradient_y[y][x]])

        # normalize z to length 30
        z_len = np.sqrt(z[0] ** 2 + z[1] ** 2)
        z_len = z_len if z_len != 0 else 1
        z_normalized = (z / z_len) * 30
        z = z_normalized.astype(int)

        dists.append(d)
        grads.append(-z)

    if plot3D:
        ut.plot2DArrayIn3D(IDT, contour, grads)

    IDT = cv.cvtColor(IDT, cv.COLOR_GRAY2BGR)

    # for each contour vertex, draw it on the image and its normal, gradient and distance
    for i in range(len(contour)):
        (x, y) = contour[i]
        z = grads[i]
        d = dists[i]
        n = normals[i]

        # draw normal (n)
        red = (0, 0, 255)
        cv.line(IDT, (x, y), (x+n[0], y+n[1]), color=red, thickness=2)

        # draw gradient (z)
        green = (0, 255, 0)
        cv.line(IDT, (x, y), (x+z[0], y+z[1]), color=green, thickness=2)

        # draw vertex (x,y)
        yellow = (0, 255, 255)
        cv.circle(IDT, (x, y), radius=4, color=yellow, thickness=-1)

        # draw distance (d)
        purple = (255, 0, 255)
        d_str = "%.2f" % d
        cv.putText(IDT, d_str, (x, y), fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=1,
                   color=purple, thickness=1, lineType=cv.LINE_AA)
    return IDT


def normalize_depth_image(img, threshold=0, amplify=False):
    img = img.astype('float64')
    img /= 255.0
    img = (img >= threshold) * img
    if amplify:
        # for visualization purpose:
        img = np.where(img > 0, img + 50, img)
    img = img.astype('uint8')
    return img


def prev_silhouette_buffer(prev_silhouette):
    """
    Calculate an approximation of valid silhouette buffer based on the silhouette from the previous frame.
    Usage: set pixels out of the buffer as background pixels:
        silhouette[buffer != SILH_PIXEL] = BG_PIXEL
    Notice that it's a bit tricky because when the person moves fast, (some of) the real silhouette might be found out
    of this buffer, and therefore be mistakenly classified as background. When using this in real time, the silhouette
    buffer gets "fixed". The bigger the kernel is, the quicker it gets fixed. But bigger kernel also means less
    accurate buffer.
    Also notice: if the person moves *too* fast, the buffer might disappear, and then we would reset it.
    """
    global out_of_buffer
    sqrt_factor = 2
    x = int(sqrt_factor * math.sqrt(out_of_buffer))
    x = min(max(x, 5), 200)                 # make sure 5 < x < 200
    kernel = (x, x)
    # kernel = (5,5)

    # flip silhouette pixels and background pixels so we can do smoothing (filter)
    flipped = np.where(prev_silhouette == SILH_PIXEL, BG_PIXEL, SILH_PIXEL)

    # smoothing
    filtered = ut.img_get_box_filter(flipped, kernel)

    # flip again to get the buffer
    buffer = np.where(filtered == 0, BG_PIXEL, SILH_PIXEL)

    # check if buffer is empty (meaning, empty of SILH_PIXEL / filled only with BG_PIXEL)
    if np.array_equal(buffer, full_background):
        # reset buffer
        buffer = np.copy(full_silhouette)

    # display result
    buffer = cv.convertScaleAbs(buffer)
    buffer_display = cv.normalize(
        buffer, None, 255, 0, cv.NORM_MINMAX, cv.CV_8UC1)
    ut.imgDisplayCV(buffer_display, 'previous silhouette buffer')

    return buffer


def update_out_of_buffer(new_val):
    print(new_val)
    global out_of_buffer
    if new_val > out_of_buffer:
        out_of_buffer = new_val
    else:
        p = 0.8
        out_of_buffer = int(p * out_of_buffer + (1-p) * new_val)
    return


def get_img_silhouette(normalized, prev_silhouette=[], bbox=[]):
    binary = np.asanyarray(normalized)
    # we consider pixels with depth > 0 as background
    binary = np.where(binary > 0, BG_PIXEL, SILH_PIXEL)
    binary = binary.astype('uint8')

    if len(bbox) == 4:
        # set pixels out of the bounding box as background pixels
        (y0, y1, x0, x1) = [int(p) for p in bbox]
        binary[:y0] = BG_PIXEL
        binary[y1:] = BG_PIXEL
        binary[:, :x0] = BG_PIXEL
        binary[:, x1:] = BG_PIXEL

    # calculate the silhouette
    x_shift = 2
    y_shift = 2
    shift = ut.shiftImg(binary, x_shift, y_shift)
    plus = binary + shift
    # we consider pixels with value 1 as the silhouette. set silhouette pixels to 0:
    silhouette = np.where(plus == 1, SILH_PIXEL, BG_PIXEL)

    # set y_shift columns and x_shift row in the image as background:
    silhouette[:y_shift] = BG_PIXEL
    silhouette[:, :x_shift] = BG_PIXEL

    if len(prev_silhouette) != 0:
        buffer = prev_silhouette_buffer(prev_silhouette)

        current_out_of_buffer = np.count_nonzero(
            silhouette[buffer == BG_PIXEL] == SILH_PIXEL)
        update_out_of_buffer(current_out_of_buffer)

        # for each pixel that is classified as background in buffer, set it as background
        # also in silhouette
        silhouette[buffer == BG_PIXEL] = BG_PIXEL

    # for visualization:
    silhouette = np.where(silhouette == BG_PIXEL, 255, SILH_PIXEL)

    silhouette = silhouette.astype('uint8')
    return silhouette


def test_silhouette_aux(depth_image, prev_silhouette=[], threshold=0, amplify=True):
    normalized = normalize_depth_image(depth_image, threshold, amplify)
    silhouette = get_img_silhouette(normalized, prev_silhouette)

    # display results
    result = np.hstack((normalized, silhouette))
    title_result = 'result: normalized, silhouette with threshold ' + \
        str(threshold)
    ut.imgDisplayCV(result, title_result)
    return silhouette


def test_silhouette(depth_image):
    test_silhouette_aux(depth_image)
    # test_silhouette_aux(depth_image, threshold=5)
    # test_silhouette_aux(depth_image, threshold=10)
    return


def test_silhouette_with_prev_frame_buffer(depth_image, prev_silhouette):
    # silhouette = test_silhouette_aux(depth_image, prev_silhouette)
    silhouette = test_silhouette_aux(depth_image, prev_silhouette, threshold=5)
    # silhouette = test_silhouette_aux(depth_image, prev_silhouette, threshold=10)
    return silhouette


def test_IDT_aux(depth_image, color_image, prev_silhouette, use_bbox=False, draw_contour=True,
                 threshold=0, real_time=True):
    normalized = normalize_depth_image(depth_image, threshold, amplify=True)

    bbox = []
    if use_bbox:
        color_image = np.expand_dims(color_image, axis=0)
        (frames_idx, bbox) = ut.bbox(color_image)
        if 0 not in frames_idx:
            # no person was detected in the frame
            bbox = []
        else:
            bbox = bbox[0]

    silhouette = get_img_silhouette(normalized, prev_silhouette, bbox)
    IDT = ut.getIDT(silhouette)

    if draw_contour:
        contour = ut.createCircleContour(IMG_WIDTH, IMG_HEIGHT)
        IDT_gradient = ut.getImgGradient(IDT)
        normals = ut.calc_normal(contour, normalize=False).astype(int)
        IDT_BGR = draw_contour_on_IDT(IDT, contour, normals, IDT_gradient)

        # convert IDT to int so that we can display it with cv.imshow
        IDT_BGR = cv.convertScaleAbs(IDT_BGR)

        silhouette_BGR = cv.cvtColor(silhouette, cv.COLOR_GRAY2BGR)
        normalized_BGR = cv.cvtColor(normalized, cv.COLOR_GRAY2BGR)
        result = np.hstack((normalized_BGR, silhouette_BGR, IDT_BGR))
    else:
        IDT = cv.convertScaleAbs(IDT)
        result = np.hstack((normalized, silhouette, IDT))

    # display result
    title_result = 'result: normalized, silhouette, IDT with threshold ' + \
        str(threshold)
    ut.imgDisplayCV(result, title_result)

    if not real_time:
        cv.waitKey(0)

    return silhouette


def test_IDT(depth_image, color_image, prev_silhouette=[], draw_contour=True, use_bbox=True):
    """
    for faster but less accurate results, set use_bbox=False
    """

    prev_silhouette = []
    # silhouette = test_IDT_aux(depth_image, color_image,
    #                         prev_silhouette, use_bbox, draw_contour, threshold=0)
    silhouette = test_IDT_aux(depth_image, color_image,
                              prev_silhouette, use_bbox, draw_contour, threshold=4)
    # silhouette = test_IDT_aux(depth_image, color_image,
    #                         prev_silhouette, use_bbox, draw_contour, threshold=10)
    return silhouette


def real_sense_tests():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, IMG_WIDTH,
                         IMG_HEIGHT, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, IMG_WIDTH,
                         IMG_HEIGHT, rs.format.z16, 30)

    # Start streaming
    pipeline.start(config)

    prev_silhouette = np.copy(full_silhouette)

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                return

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # run test:
            # test_img_gradient()
            # test_silhouette(depth_image)
            # prev_silhouette = test_silhouette_with_prev_frame_buffer(
            #     depth_image, prev_silhouette)
            prev_silhouette = test_IDT(
                depth_image, color_image, prev_silhouette)

            cv.waitKey(1)

    except Exception as e:
        print(e)
        pass

    finally:
        pipeline.stop()


def main():
    real_sense_tests()


if __name__ == '__main__':
    main()
