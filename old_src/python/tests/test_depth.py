#!/usr/bin/env python

import pyrealsense2 as rs
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def test_depth():
    try:
        # Create a context object. This object owns the handles to all connected realsense devices
        pipeline = rs.pipeline()
        pipeline.start()
        while True:
            # This call waits until a new coherent set of frames is available on a device
            # Calls to get_frame_data(...) and get_frame_timestamp(...) on a device will return stable values until wait_for_frames(...) is called
            frames = pipeline.wait_for_frames()
            depth = frames.get_depth_frame()
            if not depth: continue
            
            depth_data = depth.as_frame().get_data()
            np_image = np.asanyarray(depth_data)
            print(np_image)

    except Exception as e:
        print(e)
        pass



def test_opencv(color=True):
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            colormap = cv.COLORMAP_JET if color else cv.COLORMAP_BONE
            depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), colormap)

            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))

            # Show images
            cv.namedWindow('RealSense', cv.WINDOW_AUTOSIZE)
            cv.namedWindow('depth_image', cv.WINDOW_AUTOSIZE)
            cv.imshow('RealSense', images)
            cv.imshow('depth_image', depth_image)
            cv.waitKey(1)
    
    except Exception as e:
        print(e)
        pass

    finally:
        pipeline.stop()


# def plot_binary_img(img, title):
#     plt.title(title)
#     for i in range(len(img)):
#         if i%2 != 0: continue
#         for j in range(len(img[0])):
#             if j%2 != 0: continue
#             x = 1
#             # if img[i][j] != 0:
#             #     plt.scatter(i, j)


def normalize_depth_image(img):
    img = img.astype('float64')
    img /= 255.0
    img = (img >= 10)  * img
    # img = np.where(img >= 10, 100, 0)
    img = img.astype('uint8')
    return img


def main():
    # test_depth()
    # test_opencv(color=False)
    test_DT()


if __name__ == '__main__':
    main()