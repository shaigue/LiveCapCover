
import cv2

import config
from lib.image_processing.silhouette import LivecapBGS
from lib.data_utils.create_dataset import (
    extract_bounding_box,
    extract_vibe,
    visualize_bounding_boxes,
    visualize_vibe_data,
)


def convert_name_to_index():
    for i, frame in enumerate(config.frame_path.glob('*')):
        new_name = frame.parent / f'{i}{frame.suffix}'
        frame.rename(new_name)


def create_bgs():
    bgs = LivecapBGS(config.background_path)
    bgs_dir = config.original_dataset_path / 'background_subtraction'
    bgs_dir.mkdir(exist_ok=True)
    for frame in config.frame_path.glob('*'):
        p = bgs_dir / f'{frame.name}.png'
        frame = cv2.imread(str(frame), cv2.IMREAD_GRAYSCALE)
        mask = bgs.process(frame)
        cv2.imwrite(str(p), mask)


if __name__ == "__main__":
    # convert_name_to_index()
    # create_bgs()
    # extract_bounding_box(str(config.original_dataset_path), suffix='.png')
    # visualize_bounding_boxes(str(config.original_dataset_path), suffix='.png')
    # extract_vibe(str(config.original_dataset_path), suffix='.png')
    # visualize_vibe_data(str(config.original_dataset_path), suffix='.png')
    pass