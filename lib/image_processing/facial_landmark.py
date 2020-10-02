"""For extracting facial features."""
import os
from typing import List, Union
import urllib.request as urlreq

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray


class FaceLandmarkDetector:
    """Assumes only one person is being tracked."""
    def __init__(self):
        haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
        haarcascade = "assets/haarcascade_frontalface_alt2.xml"

        # check if file is in working directory
        if not os.path.isfile(haarcascade):
            urlreq.urlretrieve(haarcascade_url, haarcascade)

        self.face_detector = cv2.CascadeClassifier(haarcascade)

        LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"
        LBFmodel = "assets/lbfmodel.yaml"

        if not os.path.isfile(LBFmodel):
            urlreq.urlretrieve(LBFmodel_url, LBFmodel)

        self.landmark_detector = cv2.face.createFacemarkLBF()
        self.landmark_detector.loadModel(LBFmodel)

        # for tracking
        self.last_tracked = None
        self.streak_to_activate = 0
        self.activate_thresh = 3

    def initialize_sequence(self, init_sequence: List[ndarray]):
        faces_bbox_list = []
        # extract faces bounding box
        for img_gray in init_sequence:
            faces_bbox_list.append(self.face_detector.detectMultiScale(img_gray))

        running_faces = {}
        for faces in faces_bbox_list:
            # if the running faces are empty then just push
            n_detected = len(faces)
            n_tracking = len(running_faces)

            if len(running_faces) == 0:
                for i, face in enumerate(faces):
                    running_faces[i] = {'n_appeared': 1, 'bboxes': [face]}
                continue
            # calculate the similarity matrix, for NxM, where N is the currently tracked faces, and
            # M is the number of detected faces
            similarity_matrix = np.empty((n_tracking, n_detected))
            for track in range(n_tracking):
                for detect in range(n_detected):
                    last_bbox_tracked = running_faces[track]['bboxes'][-1]
                    similarity_matrix[track, detect] = np.linalg.norm(last_bbox_tracked - faces[detect])
            # keep records of what were matched
            track_unmatched = list(range(n_tracking))
            detect_unmatched = list(range(n_detected))

            # continue until one of the lists are empty
            while track_unmatched and detect_unmatched:
                track, detect = np.unravel_index(np.argmin(similarity_matrix), similarity_matrix.shape)
                # match 'track' to 'detect'
                running_faces[track]['n_appeared'] += 1
                running_faces[track]['bboxes'].append(faces[detect])
                # put infinity in the row and column
                similarity_matrix[track] = np.inf
                similarity_matrix[:, detect] = np.inf
                # remove them from the unmatched lists
                track_unmatched.remove(track)
                detect_unmatched.remove(detect)

            # all the detected that are unmatched will create a new track
            j = len(running_faces)
            for detect in detect_unmatched:
                running_faces[j] = {'n_appeared': 1, 'bboxes': [faces[detect]]}
                j += 1

        # now set self.last_tracked to the first boundind box that appeared the most
        max_appeared = 0
        for value in running_faces.values():
            if value['n_appeared'] > max_appeared:
                max_appeared = value['n_appeared']
                self.last_tracked = value['bboxes'][0]

    def detect(self, img_gray: ndarray) -> Union[ndarray, None]:
        """image should be in grayscale.
        returns a list of landmarks(ndarray shape (1, 68, 2),
        each for a different face in the image.
        """
        if self.last_tracked is None:
            raise RuntimeError('tracker is not initialized. call `initialize_sequence` first.')
        faces_bb = self.face_detector.detectMultiScale(img_gray)
        # if it is empty propagate the last face
        if len(faces_bb) == 0:
            # faces_bb = self.last_tracked.reshape((1, -1))
            self.streak_to_activate = self.activate_thresh
            return None
        # choose the most fitting one to last one
        most_fitting_distance = np.inf
        most_fitting_i = 0
        for i, face in enumerate(faces_bb):
            distance = np.linalg.norm(face - self.last_tracked)
            if distance < most_fitting_distance:
                most_fitting_distance = distance
                most_fitting_i = i
        faces_bb = faces_bb[most_fitting_i].reshape((1, -1))
        self.last_tracked = faces_bb[0]
        # if the distance is too far from the last frame then consider it as a new detection
        if most_fitting_distance > img_gray.shape[0] / 10:
            self.streak_to_activate = self.activate_thresh
            return None

        if self.streak_to_activate > 0:
            self.streak_to_activate -= 1
            return None

        _, landmarks = self.landmark_detector.fit(img_gray, faces_bb)
        return landmarks[0][0]


def main():
    # demo 2 - dataset running
    root = os.path.join('../../datasets', 'dance')
    # root = os.path.join('dataset', 'stretch')
    print('starting face tracking visualization...')
    frame_dir = os.path.join(root, 'frames')
    n_frames = len(os.listdir(frame_dir))
    print(f'found {n_frames} frames.')
    face_tracker = FaceLandmarkDetector()

    # initializing
    init_seq_len = 8
    init_seq = []
    for i in range(init_seq_len):
        frame_file = os.path.join(frame_dir, f'{i}.jpg')
        frame = cv2.imread(frame_file, cv2.IMREAD_GRAYSCALE)
        init_seq.append(frame)
    face_tracker.initialize_sequence(init_seq)

    for i in range(1300, n_frames):
        frame_file = os.path.join(frame_dir, f'{i}.jpg')

        frame = cv2.imread(frame_file, cv2.IMREAD_GRAYSCALE)
        face_landmarks = face_tracker.detect(frame)
        # if a face was not detected
        if face_landmarks is not None:
            # draw the circles on the image
            for x, y in face_landmarks:
                x = int(x)
                y = int(y)
                frame = cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)



        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    return
    # Demo 1 - might not be working now.
    image_path = 'abba.png'
    assert os.path.isfile(image_path)

    image_bgr = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    face_landmark_detector = FaceLandmarkDetector()
    landmarks = face_landmark_detector.detect(image_gray)
    print('number of faces discoverd: ', len(landmarks))
    x = []
    y = []
    for landmark in landmarks:
        for p in landmark[0]:
            # only a single element in this dimension
            x.append(p[0])
            y.append(p[1])
    plt.imshow(image_rgb)
    plt.scatter(x, y, s=1.5)
    plt.show()


if __name__ == '__main__':
    main()
