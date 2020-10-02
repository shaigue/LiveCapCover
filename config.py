"""Contains common configuration that are used in the entire project."""
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent
joint_angle_limits_path = project_root / 'assets/joint_angle_limits.csv'
datasets_dir_path = project_root / 'datasets'
animations_path = project_root / 'assets/animations/'
dance_dataset_path = datasets_dir_path / 'dance'

video_path = project_root / ''
assets_path = project_root / 'assets'

rigged_model_path = assets_path / 'models/basic_man_rigged.dae'
animated_model_path = assets_path / 'models/basic_man_animated.dae'
animated_model2_path = assets_path / 'models/basic_man_rigged_meatering.dae'
farm_boy_model_path = assets_path / 'models/farm_boy/model.dae'
farm_boy_texture_path = assets_path / 'models/farm_boy/diffuse.png'

# ================ Will Be used for the demo ===========================================================================
model_path = animated_model2_path
livecap_dataset_path = dance_dataset_path
fx_fy_t_path = assets_path / 'dance_calibration.npz'


t_blender_to_camera = np.array(
    [[1, 0, 0, 0],
     [0, 0, -1, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1]]
)

# model camera to world matrix
# translation is in blender coordinates
# and the axis are the
camera_to_world_matrix = np.identity(4)
image_width = 1280
image_height = 720
scale = dict(
    xscale=1.5,
    yscale=1.5,
)

# lambdas for energy:
energy_weights = dict(
    c_3d=1,
    c_2d=1e-5,
    c_silhouette=0,
    c_temporal=0.1,
    c_anatomic=0.01,
)

experiment_options = {
    'c_3d': [1],
    'c_2d': [1e-5, 1e-3],
    # 'c_silhouette': [1e-3, 1e-5],
    'c_temporal': [0.1, 0.01],
    'c_anatomic': [0.5, 0.01]
}

# extracted using blender edit mode
model_face_indices = {
    'nose': 3622,
    'lower_lip': 7507,
    'left_eye': 3777,
    'right_eye': 4110,
}

# ============================== Livecap Original =======================
original_path = assets_path / 'original'
motion_path = original_path / 'init2.motion'
character = 'mohammad'
calibration_path = original_path / (character + '.calibration')
mtl_path = original_path / (character + '.mtl')
obj_path = original_path / (character + '.obj')
skeleton_path = original_path / (character + '.skeleton')
skin_path = original_path / (character + '.skin')
segmentation_path = original_path / 'segmentation.txt'
texture_path = original_path / 'textureMap.png'
background_path = original_path / 'background'
original_dataset_path = datasets_dir_path / 'original'
frame_path = original_dataset_path / 'frame'
vibe_data_dir = project_root / 'lib/image_processing/vibe/data/vibe_data/'
vibe_chkpt_path = vibe_data_dir / 'vibe_model_w_3dpw.pth.tar'
spin_chkpt_path = vibe_data_dir / 'spin_model_checkpoint.pth.tar'
original_camera_path = assets_path / 'original_cam.npz'
intrinsic_params_path = original_path / 'intrinsic.npz'
original_scale_path = original_path / 'scale.npz'
experiments_dir = assets_path / 'experiments'
if not experiments_dir.is_dir():
    experiments_dir.mkdir()


t_livecap_to_camera = np.array([
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
])

exp1_weights = dict(
    c_3d=1,
    c_2d=1e-3,
    c_silhouette=0,
    c_temporal=0.1,
    c_anatomic=0.5,
)

exp2_weights = exp1_weights.copy()
exp2_weights['c_silhouette'] = 1e-3

# higher anatomic + temporal
exp3 = dict(
    c_3d=1,
    c_2d=1e-3,
    c_silhouette=1e-3,
    c_temporal=1,
    c_anatomic=2,
)
# no anatomic + no temporal
exp4 = dict(
    c_3d=1,
    c_2d=1e-3,
    c_silhouette=1e-3,
    c_temporal=0,
    c_anatomic=0,
)

# no 3d
exp5 = dict(
    c_3d=0,
    c_2d=1e-3,
    c_silhouette=1e-3,
    c_temporal=0.1,
    c_anatomic=0.5,
)

# no 2d, no silhouette
exp6 = dict(
    c_3d=1,
    c_2d=0,
    c_silhouette=0,
    c_temporal=0.1,
    c_anatomic=0.5,
)

experiment_weights = [exp1_weights, exp2_weights, exp3, exp4, exp5, exp6, exp2_weights]


read_livecap_model_args = [obj_path, skeleton_path, skin_path, segmentation_path, texture_path,
                           t_livecap_to_camera, True]
