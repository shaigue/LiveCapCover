from pathlib import Path

from numpy import ndarray
import numpy as np
import cv2
import skvideo.io

import config
from lib.data_utils.livecap_dataset import LiveCapDataset


def side_by_side(image1: ndarray, image2: ndarray, output_shape: tuple = None, y_trim=0, x_trim=0):
    # y_trim is float of x_trim is float then interpret them them like percents
    if isinstance(x_trim, float):
        x_trim = int(image1.shape[0] * x_trim)
    if isinstance(y_trim, float):
        y_trim = int(image1.shape[1] * y_trim)

    if x_trim > 0:
        image1 = image1[x_trim:-x_trim]
        image2 = image2[x_trim:-x_trim]
    if y_trim > 0:
        image1 = image1[:, y_trim:-y_trim]
        image2 = image2[:, y_trim:-y_trim]
    if output_shape is None:
        h = image1.shape[0]
        w = image1.shape[1]
    else:
        h, w = output_shape
    w1 = w // 2
    w2 = w - w1

    a = cv2.resize(image1, (w1, h))
    b = cv2.resize(image2, (w2, h))
    merged = np.concatenate((a, b), axis=1)
    return merged


def create_combined(exp_name: str, dataset_dir: Path, show=False):
    ds = LiveCapDataset(root=dataset_dir)
    exp_dir = config.experiments_dir / exp_name
    video_path = exp_dir / 'video.mp4'
    cap = cv2.VideoCapture(str(video_path))
    h = ds.image_height
    w = ds.image_width
    output_shape = (h // 2, w)
    writer = skvideo.io.FFmpegWriter(str(exp_dir / 'combined.mp4'))
    i = 0
    finished = False
    while not finished:
        received, opt_frame = cap.read()
        if received:
            print(i)
            # only take 30 seconds
            if i >= 30 * 30:
                break
            orig_frame = cv2.imread(str(ds.file_paths['frame'][i]))
            merged = side_by_side(orig_frame, opt_frame, output_shape, x_trim=0.1, y_trim=0.1)
            writer.writeFrame(merged[..., [2, 1, 0]])
            if show:
                cv2.imshow('merged', merged)
                if cv2.waitKey() == ord('q'):
                    break
            i += 1
        else:
            finished = True

    cap.release()
    writer.close()


if __name__ == "__main__":
    exp_names = [
        't_smoothing',
    ]
    for exp in exp_names:
        print(exp)
        create_combined(exp, config.original_dataset_path)


