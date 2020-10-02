"""
This uses VIBE model to extract the 2d and 3d key points from an image
"""
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import skvideo.io
import torch
import torch.nn.functional as F

import config
import lib.image_processing.bounding_box_tracking as bbt
from lib.image_processing.vibe.lib.models.vibe import VIBE_Demo


vibe_joints_names = [
    'OP Nose',        # 0
    'OP Neck',        # 1
    'OP RShoulder',   # 2
    'OP RElbow',      # 3
    'OP RWrist',      # 4
    'OP LShoulder',   # 5
    'OP LElbow',      # 6
    'OP LWrist',      # 7
    'OP MidHip',      # 8
    'OP RHip',        # 9
    'OP RKnee',       # 10
    'OP RAnkle',      # 11
    'OP LHip',        # 12
    'OP LKnee',       # 13
    'OP LAnkle',      # 14
    'OP REye',        # 15
    'OP LEye',        # 16
    'OP REar',        # 17
    'OP LEar',        # 18
    'OP LBigToe',     # 19
    'OP LSmallToe',   # 20
    'OP LHeel',       # 21
    'OP RBigToe',     # 22
    'OP RSmallToe',   # 23
    'OP RHeel',       # 24
    'rankle',         # 25
    'rknee',          # 26
    'rhip',           # 27
    'lhip',           # 28
    'lknee',          # 29
    'lankle',         # 30
    'rwrist',         # 31
    'relbow',         # 32
    'rshoulder',      # 33
    'lshoulder',      # 34
    'lelbow',         # 35
    'lwrist',         # 36
    'neck',           # 37
    'headtop',        # 38
    'hip',            # 39 'Pelvis (MPII)', # 39
    'thorax',         # 40 'Thorax (MPII)', # 40
    'Spine (H36M)',   # 41
    'Jaw (H36M)',     # 42
    'Head (H36M)',    # 43
    'nose',           # 44
    'leye',           # 45 'Left Eye', # 45
    'reye',           # 46 'Right Eye', # 46
    'lear',           # 47 'Left Ear', # 47
    'rear',           # 48 'Right Ear', # 48
]


def load_vibe(seqlen=16, device='cpu'):
    """
    Loads the pre-trained VIBE model.
    :param seqlen: the sequence length that VIBE will operate on
    :param device: on what device we want to run the vibe model
    """
    device = torch.device(device)

    vibe = VIBE_Demo(
        seqlen=seqlen,
        n_layers=2,
        hidden_size=1024,
        add_linear=True,
        use_residual=True,
        pretrained=config.spin_chkpt_path,
    ).to(device)

    pretrained_file = config.vibe_chkpt_path
    ckpt = torch.load(pretrained_file, map_location=device)
    vibe.load_state_dict(ckpt, strict=False)
    vibe.eval()

    return vibe


def resize_tensor(x: torch.Tensor, target_size: tuple):
    x = x.unsqueeze(0)
    x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
    return x


def create_vibe_batch(frames: torch.Tensor, bbox: ndarray, frames_idx: ndarray,
                      target_size=(224, 224)):
    frames = frames[frames_idx]
    batch = []
    for i, frame in enumerate(frames):
        frame = bbt.crop_tensor(frame, bbox[i])
        frame = resize_tensor(frame, target_size)
        batch.append(frame)
    batch = torch.cat(batch)
    return batch


def numpy_img_batch_to_torch(batch: ndarray):
    """Converts an image batch from np.array image to torch tensor.
    :param batch: np.array with shape (N,H,W,C) where the colors range from 0 - 255
    :returns torch.Tensor with shape (N,C,H,W) where the colors range from 0 - 1
    """
    return torch.from_numpy(batch).permute(0, 3, 1, 2) / 255.0


def kp_2d_to_orig(kp_2d: ndarray, bbox: ndarray) -> ndarray:
    """moves a 2d key point from [-1,1]^2 space in the cropped image, to pixels in the original image.

    :param kp_2d: np.array size [n, 2], values in [-1,1], (y,x)
    :param bbox: np.array size [4], x0,x1,y0,y1
    :returns the key points in the original image (x,y), np.array
    """
    # make sure that we do not go out of range:
    kp_2d = np.clip(kp_2d, -1.0, 1.0)
    # change from (y,x) -> (x, y)
    kp_2d = kp_2d[:, [1, 0]]
    # augment the kp: (x,y)->(x,y,1)
    ones = np.ones((kp_2d.shape[0], 1), dtype=kp_2d.dtype)
    kp_2d = np.concatenate((kp_2d, ones), axis=1)
    # calculate the transformation from [-1,1]^2 to the original image
    x0, x1, y0, y1 = bbox
    # calculate the dimentions of the cropped image
    h_cropped = x1 - x0
    w_cropped = y1 - y0
    # calculate the scale on each axis from 2 -> h and 2 -> w
    s_x = h_cropped / 2
    s_y = w_cropped / 2
    # should first move to the center of the cropped image
    t_x = h_cropped / 2
    t_y = w_cropped / 2
    # add the offset of the begining of the cropped image
    t_x += x0
    t_y += y0
    # put it all into a single transformation matrix:
    transformation = np.array([
        [s_x, 0.0, 0.0],
        [0.0, s_y, 0.0],
        [t_x, t_y, 1.0]
    ])
    kp_2d = kp_2d.dot(transformation)
    # remove the last axis
    return kp_2d[:, :2]


class KeypointExtractor:
    """This is an object, fed with np.array representing images,
    and extracts from them the 2d and 3d key points of the joints.

    NOTE: assumes only one person appears in the images.
    NOTE: best to use seqlen of 12, it works less when its fed single images
    """
    def __init__(self, seqlen=12, device='cpu'):

        self.tracker = bbt.VideoTracker(device)
        self.vibe = load_vibe(seqlen, device)
        self.device = device

    def extract_key_points(self, img_seq: ndarray):
        """receives a single image, as np.array of arbitrary size,

        :param img_seq: a sequence of images np.array, to be with values [0-255], shape (seqlen,H,W,3), RGB encoded
               length of the sequence can be less then seqlen, but still have the axis.
        :returns a dict with 5 entries, '2d' and '3d' for 2d and 3d entries appropriately.
                 each with dimensions of (seqlen, 49, 2) and (seqlen, 49, 3),
                 and 'u', 'v', 's' that are parameters for the projection matrix, as shown in 'explore_vibe.ipynb'
        """
        if not isinstance(img_seq, ndarray):
            raise TypeError('input should be np.array, got ', type(img_seq))
        if len(img_seq.shape) != 4 or img_seq.shape[3] != 3:
            raise ValueError('input should be image with (N,H,W,3) shape.')
        if img_seq.max() <= 1:
            raise ValueError('input should be encoded in [0-255] and not black image')

        with torch.no_grad():
            img_seq = numpy_img_batch_to_torch(img_seq).to(self.device)
            track_res = self.tracker.track(img_seq)
            # get the last person tracked
            try:
                _, track_res = track_res.popitem()
            except KeyError:
                raise RuntimeError('could not find any person in the video.')

            bbox = track_res['bbox']    # bounding box for each frame in 'frames_idx'
            frames_idx = track_res['frames']    # the frames the person was detected in

            # crop and resize
            vibe_in = create_vibe_batch(img_seq, bbox, frames_idx).unsqueeze(0).to(self.device)
            # run vibe on the images
            vibe_out = self.vibe(vibe_in)[-1]
            kp_2d = vibe_out['kp_2d'].numpy().squeeze(0)    # reduce the first dim, should be (1,N,49,2)->(N,49,2)
            kp_3d = vibe_out['kp_3d'].numpy().squeeze(0)    # same here ^^
            cam = vibe_out['theta'].numpy().squeeze(0)[:, :3] # camera parameters
            s, v, u = cam.T

            # move the key points to the original image coordinates
            orig_kp_2d = []
            for i, kp in enumerate(kp_2d):
                orig_kp_2d.append(kp_2d_to_orig(kp, bbox[i]))
            kp_2d = np.stack(orig_kp_2d)

            return {
                '2d': kp_2d,
                '3d': kp_3d,
                'u': u,
                'v': v,
                's': s,
            }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    video_file = osp.join('vibe/data', 'video_samples', 'workout.mp4')
    seqlen = 12

    vreader = skvideo.io.vreader(video_file)
    frames = [next(vreader) for _ in range(seqlen)]
    frames = np.stack(frames)
    print(frames.shape)

    extractor = KeypointExtractor(seqlen, device)
    kp = extractor.extract_key_points(frames)
    print('2d: \n', kp['2d'].shape)
    print('3d: \n', kp['3d'].shape)

    example_idx = 3
    example_img = frames[example_idx]
    example_kp_2d = kp['2d'][example_idx]

    plt.imshow(example_img)
    plt.scatter(example_kp_2d[:, 1], example_kp_2d[:, 0])
    plt.show()


if __name__ == "__main__":
    main()
