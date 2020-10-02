"""The anatomic angle limits will be read from a .csv file,
Containing the fields, for each joint except the root:
    joint_name min_x max_x min_y max_y min_z max_z

* angles are in angels for readability
* the default angles are corresponding to the maximum minumum values that
    can be received from scipy.spatial.transformation.Rotation.as_euler()

"""
import csv

import numpy as np
from numpy import ndarray

import config
from lib.utils.model import Model


class Datapoint:
    def __init__(self, joint_name: str = '', min_x: float = -180, max_x: float = 180, min_y: float = -180,
                 max_y: float = 180, min_z: float = -90, max_z: float = 90):
        self.joint_name = joint_name
        self.min_x = float(min_x)
        self.max_x = float(max_x)
        self.min_y = float(min_y)
        self.max_y = float(max_y)
        self.min_z = float(min_z)
        self.max_z = float(max_z)


def write_default_table():
    model = Model.from_file(config.model_path)
    with config.joint_angle_limits_path.open('w') as f:
        datapoint = Datapoint()
        field_names = list(datapoint.__dict__.keys())
        writer = csv.DictWriter(f, field_names)
        writer.writeheader()
        for joint in model.joint_list:
            if not joint.is_root:
                datapoint = Datapoint(joint.name)
                writer.writerow(datapoint.__dict__)


def load_joint_angle_limits(model: Model) -> (ndarray, ndarray):
    """Returns 2 arrays of the minimum and maximum values for each of
    the entries"""
    min_angles = np.empty((model.n_joints, 3))
    max_angles = np.empty((model.n_joints, 3))

    initial_angles = model.get_initial_pose().angles
    min_angles = initial_angles - np.radians(90)
    max_angles = initial_angles + np.radians(90)

    # for the root joint, set the angles limit to +-inf
    root_index = model.get_root_joint_index()
    min_angles[root_index] = -np.inf
    max_angles[root_index] = np.inf

    return min_angles.reshape(-1), max_angles.reshape(-1)

    # with config.joint_angle_limits_path.open('r') as f:
    #     datapoint = Datapoint()
    #     field_names = list(datapoint.__dict__.keys())
    #     reader = csv.DictReader(f)
    #     for row in reader:
    #         datapoint = Datapoint(**row)
    #         joint_index = model.get_joint_index_from_name(datapoint.joint_name)
    #         min_angles[joint_index] = [datapoint.min_x, datapoint.min_y, datapoint.min_z]
    #         max_angles[joint_index] = [datapoint.max_x, datapoint.max_y, datapoint.max_z]

    # min_angles = np.radians(min_angles.reshape(-1))
    # max_angles = np.radians(max_angles.reshape(-1))
    # return min_angles, max_angles


if __name__ == "__main__":
    # write_default_table()
    model = Model.from_file(config.model_path)
    min_angles, max_angles = load_joint_angle_limits(model)
    print('min angles: \n', min_angles)
    print('max angles: \n', max_angles)
