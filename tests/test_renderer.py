import time

import config

from lib.utils.renderer import Renderer, view_image_blocking
from lib.utils.model import Model
from lib.data_utils.livecap_dataset_adapter import LiveCapAdapter
from lib.utils.camera import Camera


print('loading model...')
MODEL = Model.from_file(config.model_path)
print('loading dataset...')
DATASET = LiveCapAdapter(config.livecap_dataset_path, MODEL, config.fx_fy_t_path)
print('loading camera...')
CAMERA = Camera(config.camera_to_world_matrix, config.image_height, config.image_width,
                fx=DATASET.fx, fy=DATASET.fy)
print('initializing model pose...')
POSE = MODEL.get_initial_pose()
POSE.root_translation = DATASET.get_initial_translation()
MODEL.apply_livecap_pose(POSE)

NUM_ITERATIONS = 2000
PRINT_EVERY = 100

BLOCKING_CFG = dict(
    mode='blocking',
    model=MODEL,
    joint_indices=DATASET.joint_indices,
)

VIDEO_CFG = dict(
    mode='video',
    model=MODEL,
    camera=CAMERA,
    filename='test_instantiation_res.mp4',
    joint_indices=DATASET.joint_indices,
    **config.scale,
)

IMAGE_CFG = dict(
    mode='image',
    model=MODEL,
    camera=CAMERA,
    joint_indices=DATASET.joint_indices,
    **config.scale,
)


def test_instantiation():
    print('test instantiation...')
    blocking_renderer = Renderer(**BLOCKING_CFG)
    video_renderer = Renderer(**VIDEO_CFG)
    image_renderer = Renderer(**IMAGE_CFG)
    return blocking_renderer, video_renderer, image_renderer


def test_blocking(renderer: Renderer):
    print('test blocking...')
    cam, image = renderer.draw_skeleton()
    view_image_blocking(image)
    cam, image = renderer.draw_model()
    view_image_blocking(image)


def test_video(renderer: Renderer):
    print('test video...')
    start = time.perf_counter()
    i = 0
    for _ in range(NUM_ITERATIONS):
        if i > 0 and i % PRINT_EVERY == 0:
            print(f'{i} out of {2 * NUM_ITERATIONS}')
        image = renderer.draw_skeleton()
        i += 1
    for _ in range(NUM_ITERATIONS):
        if i > 0 and i % PRINT_EVERY == 0:
            print(f'{i} out of {2 * NUM_ITERATIONS}')
        image = renderer.draw_model()
        i += 1
    renderer.close()
    end = time.perf_counter()
    print(f'time={(end - start)} seconds.')


def test_image(renderer: Renderer):
    print('test image...')


    print('stress test...')
    start = time.perf_counter()
    i = 0
    for _ in range(NUM_ITERATIONS):
        if i > 0 and i % PRINT_EVERY == 0:
            print(f'{i} out of {2 * NUM_ITERATIONS}')
        image = renderer.draw_skeleton()
        i += 1
    for _ in range(NUM_ITERATIONS):
        if i > 0 and i % PRINT_EVERY == 0:
            print(f'{i} out of {2 * NUM_ITERATIONS}')
        image = renderer.draw_model()
        i += 1

    end = time.perf_counter()
    print(f'time={(end - start)} seconds.')

    image = renderer.draw_skeleton()
    view_image_blocking(image)
    image = renderer.draw_model()
    view_image_blocking(image)
    image = renderer.draw_model_silhouette()
    view_image_blocking(image)


def run_all():
    print('running all tests...')
    blocking_renderer, video_renderer, image_renderer = test_instantiation()
    test_video(video_renderer)
    test_image(image_renderer)
    test_blocking(blocking_renderer)


if __name__ == "__main__":
    run_all()
