"""For testing the livecap dataset"""
import config
from lib.data_utils.livecap_dataset import LiveCapDataset
from lib.utils.renderer import view_image_blocking, draw_pixels_on_image


def test_subset_loading():
    ds = LiveCapDataset(config.livecap_dataset_path, ['vibe', 'bbox'])
    indices = [10, 20, 100, 500]
    for i in indices:
        item = ds[i]
        for k, v in item.__dict__.items():
            print(f"{k}: {type(v)}")


def test_kp_2d_are_pixels():
    ds = LiveCapDataset(config.livecap_dataset_path)
    indices = [10, 20, 100, 500]
    for i in indices:
        item = ds[i]
        frame = item.frame
        kp_2d = item.vibe.kp_2d
        draw_pixels_on_image(frame, kp_2d)
        view_image_blocking(frame, "kp_2d on frame")


if __name__ == "__main__":
    test_subset_loading()
    test_kp_2d_are_pixels()
