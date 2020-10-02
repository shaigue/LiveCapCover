"""visually test the adapter."""
import config
from lib.image_processing.image_distance_transform import image_distance_transform
from lib.utils.renderer import view_image_blocking, draw_pixels_on_image, Renderer, RenderMode

from lib.data_utils.livecap_dataset_adapter import LiveCapAdapter
from lib.utils.model import Model
import numpy as np

model = Model.from_file(config.model_path)
dataset = LiveCapAdapter(config.livecap_dataset_path, model, config.fx_fy_t_path)
sample_indices = [10, 300, 367, 989]
renderer = Renderer(RenderMode.blocking, model, dataset.image_h, dataset.image_w, show_axes=True,
                    joint_indices=dataset.joint_indices, **config.scale)


def test_idt():
    for i in sample_indices:
        entry = dataset[i]
        silhouette = entry.silhouette
        view_image_blocking(silhouette, 'silhouette')
        idt = image_distance_transform(silhouette)
        print('idt stats: ', idt.shape, idt.dtype, idt.max())
        view_image_blocking(idt, 'idt')


def test_keypoints():
    print(f'the kp indices {dataset.kp_indices.shape}, {dataset.kp_indices}')
    print(f'the joints indices {dataset.joint_indices.shape}, {dataset.joint_indices}')
    for i in sample_indices:
        entry = dataset[i]
        print(f'kp2d shape: {entry.kp_2d.shape}')
        image = np.zeros_like(entry.frame)
        draw_pixels_on_image(image, entry.kp_2d)
        view_image_blocking(image)
        print(f'kp3d shape: {entry.kp_3d.shape}')
        renderer.draw_skeleton(entry.kp_3d, show_both=True)


def test_translation():
    for i in sample_indices:
        item = dataset[i]
        renderer.draw_debug_skeleton(item.kp_3d,
                                     item.kp_3d + item.kp_3d_translation,
                                     model.get_p3d()[dataset.joint_indices])


if __name__ == "__main__":
    test_idt()
    test_keypoints()
    test_translation()
