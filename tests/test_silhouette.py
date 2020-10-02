import numpy as np
import cv2

from lib.image_processing.image_distance_transform import image_distance_transform, get_silhouette_idt_grad_from_mask
from lib.skeletal_pose_estimation.optimization2 import load_optimization_settings
from lib.utils.renderer import view_image_blocking
from lib.data_utils.livecap_dataset import LiveCapDataset
import config

# model, dataset, camera, renderer, initial_pose = load_optimization_settings()
# model.apply_pose_vector(initial_pose)
#
# image = renderer.draw_model()
# model_fg_mask = (image == renderer.model_color).all(axis=2).astype(np.uint8) * 255
# model_silhouette = cv2.Laplacian(model_fg_mask, cv2.CV_8U)
# dx, dy = cv2.spatialGradient(model_fg_mask)
# dx = -dx
# dy = -dy
# view_image_blocking(model_fg_mask)
# view_image_blocking(model_silhouette)
#
#
# def draw_grad(grad):
#     negative = np.where(grad < 0, 255, 0).astype('uint8')
#     view_image_blocking(negative)
#     positive = np.where(grad > 0, 255, 0).astype('uint8')
#     view_image_blocking(positive)
#
# draw_grad(nx)
# draw_grad(ny)
#
#
# exit()

dataset = LiveCapDataset(root=config.original_dataset_path)
entry = dataset[0]
directory = config.assets_path / 'idt_visualization'
directory.mkdir(exist_ok=True)
cv2.imwrite(str(directory / 'frame.png'), entry.frame[..., [2, 1, 0]])
cv2.imwrite(str(directory / 'fgmask.png'), entry.silhouette)

silhouette, idt, grad, = get_silhouette_idt_grad_from_mask(entry.silhouette > 0)
cv2.imwrite(str(directory / 'silhouette.png'), (silhouette * 255).astype(np.uint8))
cv2.imwrite(str(directory / 'idt.png'), ((np.log2(idt + 1.0) / np.log2(idt.max() + 1.0)) * 255).astype(np.uint8))
#
# fg_mask = dataset[0].silhouette
# convert to binary image
# fg_mask = np.where(fg_mask > 0, 255, 0).astype(np.uint8)
# silhouette = cv2.Laplacian(fg_mask, cv2.CV_8U)
# make the image binary and flip > 0 = 0, ==0 = 1
# pre_idt = np.where(silhouette == 255, 0, 255).astype(np.uint8)
# idt = image_distance_transform(pre_idt)
# normalize to 0-255
# idt = ((idt / idt.max()) * 255).astype(np.uint8)
#
# dx, dy = cv2.spatialGradient(idt)
#
# view_image_blocking(fg_mask)
# view_image_blocking(silhouette)
# view_image_blocking(pre_idt)
# view_image_blocking(idt)
# view_image_blocking(dx)
# view_image_blocking(dy)
#