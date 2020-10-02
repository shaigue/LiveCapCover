"""Module for running the livecap optimization."""
from scipy.optimize import least_squares, minimize
from typing import List, Dict

from lib.utils.model import Model, LivecapPose
from lib.data_utils.livecap_dataset_adapter import LiveCapAdapter, AdapterEntry
from lib.skeletal_pose_estimation.energy import Energy
from lib.utils.camera import Camera
from lib.utils.renderer import Renderer


def frame_optimization(entry: AdapterEntry, prev_pose: LivecapPose, energy: Energy,
                       scalar: bool = False) -> LivecapPose:
    """Function for a single frame in the optimization.

    :returns the optimization result for the current frame.
    """
    debug = True
    # prepossessing of the input
    prev_vector = prev_pose.to_vector()
    # run the optimization until it converges
    if scalar:
        optimization_result = minimize(energy.energy_pose, prev_vector, args=(entry,),)
    else:
        optimization_result = least_squares(energy.energy_pose, prev_vector, args=(entry,),
                                            method='lm', max_nfev=900, verbose=2)

    energy.set_debug(debug)
    energy.energy_pose(prev_vector, entry)
    energy.energy_pose(optimization_result.x, entry)
    energy.set_debug(False)

    energy.frame_end()
    # postprocessing of the output
    pose_vector = optimization_result.x
    return LivecapPose.from_vector(pose_vector)


def video_optimization(dataset: LiveCapAdapter, model: Model, camera: Camera, renderer: Renderer,
                       energy_weights: Dict) -> List[LivecapPose]:
    """Optimization for the entire video

    :returns a list with an optimization result for each frame.
    """
    # initialization
    initial_pose = model.get_initial_pose()
    initial_pose.root_translation = dataset.get_initial_translation()
    results = [initial_pose]
    scalar = False
    debug = True
    print('\n\n\n\nestimating pose in video with the following energy weights:\n' + str(energy_weights))

    energy = Energy(model, camera, dataset.joint_indices, renderer, **energy_weights,
                    return_scalar=scalar, debug=debug)
    # iterate through all of the frames
    for i in range(len(dataset)):
        print(f'\nestimating pose in frame {i}...')
        entry = dataset[i]
        result = frame_optimization(entry, results[-1], energy, scalar)
        results.append(result)

    # finalize and return the results
    return results
