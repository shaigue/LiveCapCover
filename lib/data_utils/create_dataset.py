"""Creates the dataset from the input images."""
import os
import pickle
import logging
import shutil
import argparse
from typing import Callable, List, Tuple

import cv2
import numpy as np
from numpy import ndarray
from PIL import Image
import torchvision.transforms.functional as TF
import torch
from torch import Tensor

from lib.image_processing.bounding_box_tracking import VideoTracker, bbox_to_indices
from lib.image_processing.extract_key_points import load_vibe
from lib.image_processing.vibe_transformations import kp_to_orig_image
from lib.image_processing.facial_landmark import FaceLandmarkDetector
from lib.image_processing.silhouette import SilhouetteExtractor
from lib.utils.utils import read_image_rgb
import config


logging.basicConfig(filename='create_dataset.log', filemode='w',
                    level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

# =========================== Utilities ============================================



def patch_small_person(frame: ndarray) -> ndarray:
    """For the stretch video, remove the small person on the top left."""
    frame[:300, :300, :] = 255
    frame[:100, :555, :] = 255
    return frame


def video_to_images(video_path: str, save_path: str, apply_fn: Callable[[ndarray], ndarray] = None, show: bool = False,
                    save: bool = True, suffix='.jpg') -> None:
    cap = cv2.VideoCapture(video_path)
    frame_dir = os.path.join(save_path, 'frame')
    i = 0

    if save:
        if os.path.isdir(frame_dir):
            shutil.rmtree(frame_dir)
        os.makedirs(frame_dir, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        if apply_fn is not None:
            frame = apply_fn(frame)

        if show:
            cv2.imshow('frame', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        if save:
            frame_path = os.path.join(frame_dir, f'{i}{suffix}')
            cv2.imwrite(frame_path, frame)

        i += 1

    cap.release()
    cv2.destroyAllWindows()


def paths_list_to_vibe_input(data_points: List[Tuple[str, str]]) -> Tensor:
    """Takes a list of tuples [(path to frame, path to bbox),...], and converts it into a single tensor,
    with the shape: 1xNxDxD where N is the length of the list, D is a dimension size that vibe is trained on.
    """
    if len(data_points) == 0:
        raise ValueError('got empty list.')

    data = []
    vibe_dim = 224
    for frame_path, bbox_path in data_points:
        with open(bbox_path, 'rb') as f:
            bbox = pickle.load(f)
        frame = read_image_rgb(frame_path)
        w, h = frame.size
        x0, x1, y0, y1 = bbox_to_indices(bbox, (h, w))
        frame = frame.crop((y0, x0, y1, x1)).resize((vibe_dim, vibe_dim))
        frame = TF.to_tensor(frame)
        data.append(frame)
    return torch.stack(data).unsqueeze(0)


def vibe_data_generator(root: str, seqlen: int = 16, suffix='.jpg'):
    """Generates batches for running VIBE on cpu on the desired sequence length.
        1. finds the image + bounding box
        2. crops the image according to the bounding box
        3. resize the image to 224 * 224
        4. transform it into torch tensor batch, with the correct dimensions
    """
    frame_dir = os.path.join(root, 'frame')
    bbox_dir = os.path.join(root, 'bbox')
    n_frame = len(os.listdir(frame_dir))

    # list of tuples of paths to images and bounding boxes paths
    data_points = []
    for i in range(n_frame):
        frame_path = os.path.join(frame_dir, f'{i}{suffix}')
        bbox_path = os.path.join(bbox_dir, f'{i}.pkl')
        data_points.append((frame_path, bbox_path))
        if len(data_points) == seqlen:
            batch = paths_list_to_vibe_input(data_points)
            data_points = []
            yield batch

    if len(data_points) != 0:
        yield paths_list_to_vibe_input(data_points)

# =========================== Extract ====================================================


def square_bbox(bbox: ndarray) -> ndarray:
    # make the bbox square, take the max out of h,w and add it in both directions to the center of the bbox
    x0, x1, y0, y1 = bbox
    h = x1 - x0
    w = y1 - y0
    center_x = (x0 + x1) / 2
    center_y = (y0 + y1) / 2
    offset = max(h, w) / 2
    x0 = center_x - offset
    x1 = center_x + offset
    y0 = center_y - offset
    y1 = center_y + offset
    return np.array([x0, x1, y0, y1])


def extract_bounding_box(root: str, suffix='.jpg') -> None:
    """Extracts the bounding box of all the images in the folder"""
    logging.info(f'extracting bounding box in {root}.')

    frame_dir = os.path.join(root, 'frame')
    bbox_dir = os.path.join(root, 'bbox')
    shutil.rmtree(bbox_dir, ignore_errors=True)
    os.makedirs(bbox_dir, exist_ok=True)

    n_frame = len(os.listdir(frame_dir))

    logging.info(f'detected {n_frame} frame.')
    counter = 0

    tracker = VideoTracker()
    # Note: in some of the frame, there might be no human detected at all and an object will not get created.

    for i in range(n_frame):
        frame_path = os.path.join(frame_dir, f'{i}{suffix}')
        image = read_image_rgb(frame_path)
        image = TF.to_tensor(image).unsqueeze(0)
        track_result = tracker.track(image)

        # in case the detector cannot detect a person in the current frame
        try:
            _, track_result = track_result.popitem()
        except KeyError:
            logging.info(f'could not find person in frame {i}')
            continue

        bbox = track_result['bbox'][0]
        bbox = square_bbox(bbox)

        bbox_path = os.path.join(bbox_dir, f'{i}.pkl')
        with open(bbox_path, 'wb') as f:
            pickle.dump(bbox, f)

        counter += 1
        if i % 10 == 0 and i != 0:
            logging.info(f'{i} out of {n_frame}.')

    logging.info(f'finished. got {counter} out of {n_frame} bounding boxes.')


@torch.no_grad()
def extract_vibe(root: str, suffix='.jpg') -> None:
    """Extracts all the features that VIBE returns and saves them as pickle for each frame.
    Note: needs to run *AFTER* `extract_bounding_box` runs.
    """
    logging.info(f'extracting vibe features in {root}')

    vibe = load_vibe()
    vibe_dir = os.path.join(root, 'vibe')
    shutil.rmtree(vibe_dir, ignore_errors=True)
    os.makedirs(vibe_dir, exist_ok=True)

    i = 0
    for batch in vibe_data_generator(root, suffix=suffix):
        vibe_out = vibe(batch)[-1]

        n_samples = batch.shape[1]
        for j in range(n_samples):
            vibe_data = {}
            # taking the data point that corresponds to the i'th image
            for key in vibe_out.keys():
                vibe_data[key] = vibe_out[key][0, j].numpy()
            vibe_path = os.path.join(vibe_dir, f'{i}.pkl')
            with open(vibe_path, 'wb') as f:
                pickle.dump(vibe_data, f)

            i += 1

        logging.info(f'extracted {i} out of {60 * 30}')

    logging.info(f'finished. extracted {i} images.')


def extract_face(root: str, suffix='.jpg') -> None:
    """Extracts the facial key points from the images
    """
    logging.info(f'extracting faces in {root}')

    frame_dir = os.path.join(root, 'frame')
    face_detector = FaceLandmarkDetector()
    face_dir = os.path.join(root, 'face')
    shutil.rmtree(face_dir, ignore_errors=True)
    os.makedirs(face_dir, exist_ok=True)

    n_frame = len(os.listdir(frame_dir))
    total_extracted = 0
    # initializing to find the correct face
    init_seq_len = 8
    init_seq = []
    for i in range(init_seq_len):
        frame_file = os.path.join(frame_dir, f'{i}{suffix}')
        frame = cv2.imread(frame_file, cv2.IMREAD_GRAYSCALE)
        init_seq.append(frame)
    face_detector.initialize_sequence(init_seq)

    for i in range(n_frame):
        frame_path = os.path.join(frame_dir, f'{i}{suffix}')
        face_path = os.path.join(face_dir, f'{i}.pkl')

        frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        landmarks = face_detector.detect(frame)
        if landmarks is not None:
            with open(face_path, 'wb') as f:
                pickle.dump(landmarks, f)
            total_extracted += 1

        if i % 100 == 0:
            logging.info(f'{i} out of {n_frame}')

    logging.info(f'finished extracting faces. got {total_extracted} total.')


def extract_silhouette(root: str, drop_weak: bool = False, suffix='.jpg') -> None:
    """Extracts the facial key points from the images
    """
    logging.info(f'extracting silhouettes in {root}')

    frame_dir = os.path.join(root, 'frame')
    silhouette_extractor = SilhouetteExtractor()
    silhouette_dir = os.path.join(root, 'silhouette')
    shutil.rmtree(silhouette_dir, ignore_errors=True)
    os.makedirs(silhouette_dir, exist_ok=True)

    n_frame = len(os.listdir(frame_dir))

    for i in range(n_frame):
        frame_path = os.path.join(frame_dir, f'{i}{suffix}')
        silhouette_path = os.path.join(silhouette_dir, f'{i}.png')

        frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        silhouette = silhouette_extractor.process(frame, drop_weak)
        cv2.imwrite(silhouette_path, silhouette)

        if i % 100 == 0:
            logging.info(f'{i} out of {n_frame}')

    logging.info(f'finished extracting silhouettes. got {n_frame} total.')


# ============================== Visualize ============================================

def visualize_bounding_boxes(root: str, suffix='.jpg') -> None:
    """Visualizes the bounding box extracted."""
    print('starting visualize bounding box...')

    frame_dir = os.path.join(root, 'frame')
    bbox_dir = os.path.join(root, 'bbox')
    n_frame = len(os.listdir(frame_dir))

    for i in range(n_frame):
        frame_path = os.path.join(frame_dir, f'{i}{suffix}')
        bbox_path = os.path.join(bbox_dir, f'{i}.pkl')

        image = cv2.imread(frame_path)
        with open(bbox_path, 'rb') as f:
            bbox = pickle.load(f)
        x0, x1, y0, y1 = bbox_to_indices(bbox, image.shape[:2])

        image = cv2.rectangle(image, (y0, x0), (y1, x1), color=(0, 0, 255), thickness=2)
        cv2.imshow('frame', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def visualize_vibe_data_creation(root: str):
    print('starting vibe data set creation visualization...')
    counter = 0
    exit_loop = False
    for batch in vibe_data_generator(root):
        batch = batch[0]
        for frame in batch:
            frame = frame.permute(1, 2, 0).numpy()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow('cropped frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit_loop = True
                break
            counter += 1
        if exit_loop:
            break
    cv2.destroyAllWindows()
    print(f'{counter} images were iterated through.')


def visualize_vibe_data(root: str, suffix='.jpg') -> None:
    """Visualizes the vibe data on the original frame.
    1. draw the 2d points.
    2. draw the 3d points.
    """
    print('starting vibe data visualization...')
    frame_dir = os.path.join(root, 'frame')
    bbox_dir = os.path.join(root, 'bbox')
    vibe_dir = os.path.join(root, 'vibe')

    n_frame = len(os.listdir(frame_dir))
    print(f'found {n_frame} frame.')

    for i in range(n_frame):
        frame_file = os.path.join(frame_dir, f'{i}{suffix}')
        bbox_file = os.path.join(bbox_dir, f'{i}.pkl')
        vibe_file = os.path.join(vibe_dir, f'{i}.pkl')

        frame = cv2.imread(frame_file)
        with open(bbox_file, 'rb') as f:
            bbox = pickle.load(f)
        with open(vibe_file, 'rb') as f:
            vibe = pickle.load(f)

        kp_2d = vibe['kp_2d']
        kp_3d = vibe['kp_3d']
        cam_params = vibe['theta'][:3]
        h, w, _ = frame.shape

        kp_2d_pix = kp_to_orig_image(h, w, cam_params, bbox, kp_2d)
        kp_3d_pix = kp_to_orig_image(h, w, cam_params, bbox, kp_3d)

        # draw the circles on the image
        for x, y in kp_3d_pix:
            x = int(x)
            y = int(y)
            frame = cv2.circle(frame, (y, x), 3, (255, 0, 0), -1)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def visualize_face(root: str, suffix='.jpg') -> None:
    """Visualize the faces extracted from the images."""
    print(f'visualizing faces extracted in {root}')

    frame_dir = os.path.join(root, 'frame')
    face_dir = os.path.join(root, 'face')

    n_frame = len(os.listdir(frame_dir))
    count = 0
    for i in range(n_frame):
        frame_path = os.path.join(frame_dir, f'{i}{suffix}')
        face_path = os.path.join(face_dir, f'{i}.pkl')

        frame = cv2.imread(frame_path)
        if os.path.isfile(face_path):

            with open(face_path, 'rb') as f:
                landmarks = pickle.load(f)

            for x, y in landmarks:
                x = int(x)
                y = int(y)
                frame = cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
            count += 1

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print(f'finished visualizing faces. got {count} frame total.')


def visualize_silhouette(root: str, suffix='.jpg') -> None:
    """Visualize the faces extracted from the images."""
    print(f'visualizing silhouette extracted in {root}')

    frame_dir = os.path.join(root, 'frame')
    silhouette_dir = os.path.join(root, 'silhouette')

    n_frame = len(os.listdir(frame_dir))
    for i in range(n_frame):
        frame_path = os.path.join(frame_dir, f'{i}{suffix}')
        silhouette_path = os.path.join(silhouette_dir, f'{i}.png')

        frame = cv2.imread(frame_path)
        silhouette = cv2.imread(silhouette_path)

        # idt = np.log(idt + 1) / np.log(idt.max() + 1)
        cv2.imshow('frame', frame)
        cv2.imshow('silhouette', silhouette)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print(f'finished visualizing silhouette. got {n_frame} frame total.')


def main(args):
    # create the dataset's path from the name
    root = str(config.datasets_dir_path / args.name)
    if args.action == 'create':
        if args.video_path is None:
            raise argparse.ArgumentError('got action "create" without video [--video_path=<video_path>] to process.')
        # Split the video into frame
        if args.all or args.frames:
            video_to_images(args.video_path, root)
        if args.all or args.bbox:
            extract_bounding_box(root)
        if args.all or args.vibe:
            extract_vibe(root)
        if args.all or args.face:
            extract_face(root)
        if args.all or args.silhouette:
            drop_weak = False
            extract_silhouette(root, drop_weak)

    elif args.action == 'visualize':
        if args.all or args.frames:
            pass
        if args.all or args.bbox:
            visualize_bounding_boxes(root)
        if args.all or args.vibe:
            visualize_vibe_data(root)
        if args.all or args.face:
            visualize_face(root)
        if args.all or args.silhouette:
            visualize_silhouette(root)

    else:
        assert False, 'should not get here.'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create LiveCap dataset from video.')
    parser.add_argument('--video_path', type=str, required=False, help='Video path to be processed.', default=None)
    parser.add_argument('--name', type=str, required=True, help='The subdirectory name, that we will visualize / '
                                                                'create')
    parser.add_argument('--action', choices=['create', 'visualize'], required=True,
                        help="choose either to 'create' the data set, of to 'visualize' the created dataset.")
    parser.add_argument('--bbox', action='store_true', help='if you want to create/visualize bounding boxes')
    parser.add_argument('--vibe', action='store_true', help='if you want to create/visualize vibe features')
    parser.add_argument('--face', action='store_true', help='if you want to create/visualize facial landmarks')
    parser.add_argument('--silhouette', action='store_true', help='if you want to create/visualize silhouette')
    parser.add_argument('--frames', action='store_true', help='if you want to create the frames')
    parser.add_argument('--all', action='store_true', help='if you want to run all the processes')
    # args = parser.parse_args()
    main(parser.parse_args())
