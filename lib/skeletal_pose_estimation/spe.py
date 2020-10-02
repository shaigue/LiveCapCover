import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import logging

from scipy.optimize import least_squares

from lib.utils.raw_model_data import Joint
from lib.skeletal_pose_estimation.energy import energy
from lib.LBS.Animation import KeyFrame, Animation
from lib.LBS.Model import Model
from lib.LBS.LBS import ModelRenderer
from lib.data_utils.livecap_dataset_adapter import LiveCapAdapter, AdapterEntry


def estimate_pose(model: Model, dataset_path: Path, save_path: Path = None) -> Animation:
    '''
    1. Runs optimization etc...
    2. Returns animation
    '''

    logging.info('estimating pose from dateset: ' + str(dataset_path))

    dataset = LiveCapAdapter(dataset_path, model.root_joint)
    frame_height, frame_width, _ = dataset[0].frame.shape
    bind_pose_as_optimization_array = model.root_joint.bind_pose_to_optimization_array()
    prev_opt = bind_pose_as_optimization_array

    key_frames = []
    with ModelRenderer(model, window_width=frame_width, window_height=frame_height) as renderer:
        timestamp = 0.0
        for i, frame_data in enumerate(dataset):
            logging.info(f'estimating pose in frame {i}')
            timestamp = i * (1/30)
            if i == 150:
                break
            if i % 2 == 1:
                continue
            opt = _estimate_pose_in_frame(model.root_joint, frame_data, renderer, prev_opt)
            kf = KeyFrame(model.root_joint.optimization_array_to_pose(opt), timestamp)
            key_frames.append(kf)
            prev_opt = opt

    animation = Animation(key_frames)

    if save_path is None:
        logging.info('no save path given. not saving animation!')
    else:
        save_animation(animation, save_path)

    return animation


debug = True
# debug = False


def _estimate_pose_in_frame(root_joint: Joint, frame_data: AdapterEntry, renderer: ModelRenderer, prev_opt: np.ndarray):
    opt = least_squares(fun=energy, x0=prev_opt, method='lm', args=(root_joint, frame_data, renderer, prev_opt), max_nfev=450, verbose=2)
    energy(prev_opt, root_joint, frame_data, renderer, prev_opt, verbose=debug, log=True)
    energy(opt.x, root_joint, frame_data, renderer, prev_opt, verbose=debug, log=True)
    return opt.x


def save_animation(animation: Animation, save_path: Path):
    save_path = save_path / 'animation_' / datetime.now().strftime("%y_%m_%d_%H_%M") / '.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(animation, f)
    return


def load_animation(animation_path: Path):
    with open(animation_path, 'rb') as f:
        animation = pickle.load(f)
    return animation
