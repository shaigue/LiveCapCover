from lib.IO import farm_boy_reader
from lib.IO import read_collada
from lib.utils.raw_model_data import RawModelData
from lib.skeletal_pose_estimation import spe
from lib.LBS.Model import Model
from lib.LBS.Animation import Animation
from lib.LBS.LBS import render

import config
import logging


def main():
    raw_model_data: RawModelData = read_collada.read_collada_file(config.animated_model2_path)
    model: Model = Model.from_raw_model_data(raw_model_data)
    animation: Animation = spe.estimate_pose(model, config.livecap_dataset_path)
    # spe.save_animation(animation, config.animations_path)
    render(model, animation)


def load_estimation():
    raw_model_data: RawModelData = read_collada.read_collada_file(config.animated_model2_path)
    model: Model = Model.from_raw_model_data(raw_model_data)
    animation = spe.load_animation(config.animations_path / 'animation.pkl')
    # render(model, animation)
    return


def farm_boy():
    model: Model = farm_boy_reader.read(config.farm_boy_model_path)
    render(model, model.animation)
    return


if __name__ == '__main__':
    logging.info("********* Main started *************")
    main()
    # load_estimation()
    # farm_boy()
