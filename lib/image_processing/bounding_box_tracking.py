"""
    cut = img[:, x0:x1, y0:y1] - for tensor,

    cut = img[x0:x1, y0:y1] - for ndarry
"""
import torch
from yolov3.yolo import YOLOv3
from multi_person_tracker import Sort
import numpy as np
import os.path as osp
import skvideo.io
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt


class VideoTracker:
    def __init__(self, device='cpu', img_size=608, thresh=0.7):
        self.detector = YOLOv3(device=device, img_size=img_size,
                               person_detector=True, video=True,
                               return_dict=True)
        self.tracker = Sort()
        self.tresh = thresh
        self.device = device

    def track(self, frames: torch.Tensor):
        """
        frames: torch.Tensor, with dimentions NxCxHxW
        output: dict, ordered:
            {
                person_id:
                {
                    'bbox': np.array [[x0, x1, y0, y1], ..., ] - bounding box for each frame that the
                        person appeared in.
                    'frames': np.array [i1,i2,...] - those are the frames that this person appeared in.
                }
                ...
            }
        """
        with torch.no_grad():
            frames = frames.to(self.device)
            detector_res = self.detector(frames)

            tracks = []
            for res in detector_res:
                bbox = res['boxes'].cpu().numpy()
                scores = res['scores'].cpu().numpy()[..., None]
                detections = np.hstack((bbox, scores))
                detections = detections[scores[:, 0] > self.tresh]

                if detections.shape[0] > 0:
                    detections = self.tracker.update(detections)
                else:
                    detections = np.empty((0, 5))
                tracks.append(detections)

            return self.tracks2dict(tracks)

    @staticmethod
    def tracks2dict(trackers):
        """
        transforms the trackers list into a dictionary by the person's Id.
        """
        people = {}

        for frame_idx, tracks in enumerate(trackers):
            for d in tracks:
                y0, x0, y1, x1, person_id = d
                person_id = int(person_id)
                bbox = np.array([x0, x1, y0, y1])

                if person_id in people.keys():
                    people[person_id]['bbox'].append(bbox)
                    people[person_id]['frames'].append(frame_idx)
                else:
                    people[person_id] = {
                        'bbox': [],
                        'frames': [],
                    }
                    people[person_id]['bbox'].append(bbox)
                    people[person_id]['frames'].append(frame_idx)

        for k in people.keys():
            people[k]['bbox'] = np.array(people[k]['bbox']).reshape((len(people[k]['bbox']), 4))
            people[k]['frames'] = np.array(people[k]['frames'])

        return people


def tensor_imshow(img_t: torch.Tensor):
    img_t = img_t.permute(1, 2, 0)
    plt.imshow(img_t)


def bbox_to_indices(bbox, img_shape):
    x0, x1, y0, y1 = bbox
    x0 = max(0, int(x0))
    y0 = max(0, int(y0))
    x1 = min(img_shape[0] - 1, int(x1))
    y1 = min(img_shape[1] - 1, int(y1))
    return x0, x1, y0, y1


def draw_bb(img_t: torch.Tensor, bbox: np.array):
    """ draws a bounding box on a tensor:
    img_t: an image of size CxHxW,
    bbox: 4-vector, with [x0,x1,y0,y1]
    """
    # x is in the H dimention and y is on the W dimention
    x0, x1, y0, y1 = bbox_to_indices(bbox, img_t.shape[1:])

    img_t[0, x0:x0 + 3, y0:y1] = 1
    img_t[1, x0:x0 + 3, y0:y1] = 0
    img_t[2, x0:x0 + 3, y0:y1] = 0

    img_t[0, x1:x1 + 3, y0:y1] = 1
    img_t[1, x1:x1 + 3, y0:y1] = 0
    img_t[2, x1:x1 + 3, y0:y1] = 0

    img_t[0, x0:x1, y0:y0 + 3] = 1
    img_t[1, x0:x1, y0:y0 + 3] = 0
    img_t[2, x0:x1, y0:y0 + 3] = 0

    img_t[0, x0:x1, y1:y1 + 3] = 1
    img_t[1, x0:x1, y1:y1 + 3] = 0
    img_t[2, x0:x1, y1:y1 + 3] = 0

    return img_t


def crop_tensor(x: torch.Tensor, bbox: np.array):
    x0, x1, y0, y1 = bbox_to_indices(bbox, x.shape[1:])
    return x[:, x0:x1, y0:y1].clone()


def main():
    video_file = osp.join('vibe/data', 'video_samples', 'workout.mp4')
    frames_generator = skvideo.io.vreader(video_file)

    frame1 = next(frames_generator)
    frame2 = next(frames_generator)

    print(f'numpy images shapes: {frame1.shape}, {frame2.shape}')
    plt.imshow(np.concatenate((frame1, frame2), axis=1))
    plt.show()

    frame1_t = TF.to_tensor(frame1)
    frame2_t = TF.to_tensor(frame2)
    print(f'torch images shape: {frame1_t.shape}, {frame2_t.shape}')
    tensor_imshow(frame1_t)
    plt.show()

    frames = torch.stack((frame1_t, frame2_t))
    print(f'stacked tensor images shape: {frames.shape}')

    # BUG: some how, the id of different tracker object is non-overlapping. something global is going on here
    video_tracker = VideoTracker()
    track_res = video_tracker.track(frames)
    print('The tracking results: \n', track_res)

    bb_frames = frames.clone()
    for person_id, res in track_res.items():
        for i, frame in enumerate(res['frames']):
            draw_bb(bb_frames[frame], res['bbox'][i])

    tensor_imshow(bb_frames[0])
    plt.show()
    tensor_imshow(bb_frames[1])
    plt.show()

    cropped_frames = []
    for person_id, res in track_res.items():
        for i, frame in enumerate(res['frames']):
            cropped_frames.append(crop_bb(frames[frame], res['bbox'][i]))
    print(f'number of cropped images: {len(cropped_frames)}')

    for img_t in cropped_frames:
        tensor_imshow(img_t)
        plt.show()


if __name__ == '__main__':
    main()
