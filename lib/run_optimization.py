import itertools
from datetime import datetime

import config
from lib.skeletal_pose_estimation.optimization import video_optimization
from lib.utils.model import Model
from lib.data_utils.livecap_dataset_adapter import LiveCapAdapter
from lib.utils.camera import Camera
from lib.utils.renderer import Renderer, RenderMode
from lib.utils.utils import save_object


def init_optimization_from_config():
    model = Model.from_file(config.model_path)
    dataset = LiveCapAdapter(config.livecap_dataset_path, model, config.fx_fy_t_path)
    camera = Camera(config.camera_to_world_matrix, dataset.image_h, dataset.image_w, dataset.fx, dataset.fy)
    renderer = Renderer(RenderMode.image, model, camera=camera, joint_indices=dataset.joint_indices,
                        **config.scale)
    return model, dataset, camera, renderer


def run_default_weights_optimization():
    """Main function that runs the livecap algorithm on our model and dataset."""
    model, dataset, camera, renderer = init_optimization_from_config()

    animation = video_optimization(dataset, model, camera, renderer, config.energy_weights)
    save_object((animation, config.energy_weights), config.animations_path)


def weights_generator() -> dict:
    options = config.experiment_options
    option_keys = list(options.keys())
    options_list = [options[key] for key in option_keys]
    for item in itertools.product(*options_list):
        weights = dict(zip(option_keys, item))
        yield weights


def run_weights_experiment():
    model, dataset, camera, renderer = init_optimization_from_config()
    experiment_dir = config.animations_path / ('experiment_' + datetime.now().strftime("%y%m%d_%H%M"))

    for i, energy_weights in enumerate(weights_generator()):
        animation = video_optimization(dataset, model, camera, renderer, energy_weights)
        result = {"animation": animation, "model": model, "energy_weights": energy_weights}
        save_object(result, experiment_dir, f'optimization_{i}')


if __name__ == "__main__":
    run_weights_experiment()
