"""This is a module that takes care of rendering."""
from enum import auto, Enum
from pathlib import Path
from typing import Union
import itertools

import cv2
import numpy as np
from numpy import ndarray
import pyvista as pv
import skvideo.io

from lib.utils.camera import Camera
from lib.utils.model import Model


class Color:
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    cyan = (0, 255, 255)
    magenta = (255, 0, 255)
    yellow = (255, 255, 0)


class RenderMode(Enum):
    """Enum class for the rendering mode.

    blocking: will open a window and will block the program until the window is closed
    video: will save all the images into a video and will write it to a file in the end
    images: takes a single image and returns it without blocking
    """
    blocking = auto()
    video = auto()
    image = auto()


def find_vibe_skeleton_connectivity(skeleton_connectivity: ndarray, joint_indices: ndarray):
    all_vertices = set(np.unique(skeleton_connectivity))
    subset_vertices = set(joint_indices)
    vertices_to_remove = all_vertices - subset_vertices
    edges = set(map(lambda x: (x[0], x[1]), skeleton_connectivity))

    for v in vertices_to_remove:
        begin_with_v = set(filter(lambda x: x[0] == v, edges))
        end_with_v = set(filter(lambda x: x[1] == v, edges))
        edges = edges - end_with_v
        edges = edges - begin_with_v
        for e1, e2 in itertools.product(end_with_v, begin_with_v):
            v1 = e1[0]
            v2 = e2[1]
            edges.add((v1, v2))

    # the indicae of the joints are now changed, create the mapping and then apply
    old_to_new_vertices = {v: i for i, v in enumerate(joint_indices)}

    n_edges = len(edges)
    connectivity = np.empty((n_edges, 2), dtype=int)
    for i, (v1, v2) in enumerate(edges):
        connectivity[i] = [old_to_new_vertices[v1], old_to_new_vertices[v2]]
    return connectivity


def view_image_blocking(image: ndarray, name: str = 'image'):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    cv2.imshow(name, image)
    cv2.waitKey()
    cv2.destroyWindow(name)


def view_video_blocking(video_file: Union[Path, str]):
    if isinstance(video_file, Path):
        video_file = str(video_file)

    video = cv2.VideoCapture(video_file)
    while True:
        received, frame = video.read()
        if not received:
            break

        cv2.imshow('video', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    video.release()
    cv2.destroyWindow('video')


def draw_pixels_on_image(image: ndarray, pixels: ndarray, color: Union[tuple, str] = Color.red):
    if isinstance(color, str):
        color = Color.__dict__[color]
    pixels = np.round(pixels).astype(int)
    for i, j in pixels:
        cv2.circle(image, center=(j, i), radius=5, color=color, thickness=-1)


class Renderer:
    """Renderer class.

    Has tree possible modes:
        blocking: opens an interactive window.
            each draw function will return a tuple - (camera_pose, image) where camera pose is the
            position of the camera in the last image before the window was closed, and image is the last image.
        video: saves all the frames to a video, returns nothing on draw function
        image: renders an image.
    """

    def __init__(self, mode: Union[RenderMode, str], model: Model, image_h: int = None, image_w: int = None,
                 filename: Union[str, Path] = None, show_axes: bool = False, joint_indices: ndarray = None,
                 xscale: float = 1, yscale: float = 1, zscale: float = 1, camera: Camera = None):
        if isinstance(mode, str):
            mode = RenderMode[mode]
        if mode == RenderMode.video and filename is None:
            raise RuntimeError('filename has to be specified if mode is video')

        if mode == RenderMode.image:
            self.bg_color = np.array([0, 255, 0])
            self.model_color = np.array([0, 0, 255])

        self.camera = camera
        self.vtk_camera = None
        self.image_h = image_h
        self.image_w = image_w
        self.show_axes = show_axes
        self.mode = mode
        self.model = model
        self.xscale = xscale
        self.yscale = yscale
        self.zscale = zscale

        if self.mode == RenderMode.video:
            self.writer = skvideo.io.FFmpegWriter(str(filename))

        self.faces = self.to_pyvista_polygon(model.faces)
        self.skeleton_connectivity = self.to_pyvista_polygon(model.skeleton_connectivity)

        # make a connectivity for vibe's joints
        self.vibe_skeleton_connectivity = self.skeleton_connectivity
        if joint_indices is not None:
            transformed_skeleton_connectivity = find_vibe_skeleton_connectivity(model.skeleton_connectivity,
                                                                                joint_indices)
            self.vibe_skeleton_connectivity = self.to_pyvista_polygon(transformed_skeleton_connectivity)

        # initiialize the plotter if it is video or image
        self.plotter = None
        if self.mode in [RenderMode.video, RenderMode.image]:
            self.plotter = pv.Plotter(off_screen=True)
        if self.mode == RenderMode.image:
            self.mesh = pv.PolyData(self.model.vertices, self.faces)
            self.plotter.add_mesh(self.mesh, color=self.model_color)
            self.plotter.background_color = self.bg_color
            self.plotter.camera_position = self.camera.to_vtk_camera_pose()
            self.plotter.window_size = self.camera.image_shape
            self.plotter.set_scale(xscale=self.xscale, yscale=self.yscale, zscale=self.zscale,
                                   reset_camera=False)

    def _add_mesh(self, mesh, is_skeleton, pcolor='red', bcolor='blue'):
        if is_skeleton:
            self.plotter.add_mesh(mesh, edge_color=bcolor, show_edges=True)
            self.plotter.add_mesh(pv.PolyData(mesh.points), color=pcolor)
        else:
            self.plotter.add_mesh(mesh)



    def _plot(self, mesh1, pcolor1='red', bcolor1='blue',
              mesh2=None, pcolor2='red', bcolor2='blue',
              mesh3=None, pcolor3='red', bcolor3='blue',
              is_skeleton: bool = False, title: str = None, ):
        if self.mode == RenderMode.blocking:
            self.plotter = pv.Plotter()
        else:
            self.plotter.clear()

        self._add_mesh(mesh1, is_skeleton, pcolor1, bcolor1)
        if mesh2 is not None:
            self._add_mesh(mesh2, is_skeleton, pcolor2, bcolor2)
        if mesh3 is not None:
            self._add_mesh(mesh3, is_skeleton, pcolor3, bcolor3)

        # set the camera
        if self.camera is not None:
            self.plotter.camera_position = self.camera.to_vtk_camera_pose()
            self.plotter.window_size = self.camera.image_shape
        else:
            if self.vtk_camera is not None:
                self.plotter.camera_position = self.vtk_camera
            if self.image_w is not None and self.image_h is not None:
                self.plotter.window_size = (self.image_w, self.image_h)

        self.plotter.set_scale(xscale=self.xscale, yscale=self.yscale, zscale=self.zscale,
                          reset_camera=False)
        if self.show_axes:
            self.plotter.show_grid()
            self.plotter.show_axes()

        if title is not None:
            self.plotter.title = title

        if self.mode == RenderMode.blocking:
            return self.plotter.show(return_img=True)
        elif self.mode == RenderMode.image:
            self.plotter.background_color = self.bg_color
            return self.plotter.image
        else:  # self.mode == RenderMode.video:
            self.writer.writeFrame(self.plotter.image)

    def draw_skeleton(self, vibe_kp_3d: ndarray = None, show_both: bool = False):

        model_skeleton = pv.PolyData(self.model.get_p3d(), self.skeleton_connectivity)
        if vibe_kp_3d is not None:
            vibe_skeleton = pv.PolyData(vibe_kp_3d, self.vibe_skeleton_connectivity)
            if show_both:
                return self._plot(model_skeleton, Color.red, Color.blue,
                                  vibe_skeleton, Color.yellow, Color.green,
                                  is_skeleton=True)

            return self._plot(vibe_skeleton, Color.yellow, Color.green,
                              is_skeleton=True)
        return self._plot(model_skeleton, Color.red, Color.blue, is_skeleton=True)

    def draw_model(self, with_texture=False):
        if self.mode == RenderMode.image:
            self.mesh.points = self.model.vertices
            self.plotter.add_mesh(self.mesh, color=self.model_color)
            return self.plotter.image

        model_mesh = pv.PolyData(self.model.vertices, self.faces)
        if with_texture:
            model_mesh.t_coords = self.model.vertex_texture_coords
            texture = pv.numpy_to_texture(self.model.texture)
            model_mesh.textures[0] = texture
        return self._plot(model_mesh)

    def close(self):
        if self.mode == RenderMode.video:
            self.writer.close()

    def get_model_fg_mask(self):
        if self.mode != RenderMode.image:
            raise RuntimeError
        image = self.draw_model()
        fg_mask = (image == self.model_color).all(axis=-1)
        return fg_mask

    def draw_model_silhouette(self):
        """Draws the model's silhouette, as a binary image."""
        if self.mode != RenderMode.image:
            raise RuntimeError
        model_image = self.draw_model()
        # subtract the shifted image by one pixel, right and down
        model_image[:-1, :-1] -= model_image[1:, 1:]
        # zero the parts of the image that were not subtracted from
        model_image[-1, :] = 0
        model_image[:, -1] = 0
        # find all of the places that an image is not zero, those are the edges
        silhouette = np.any(model_image != 0, axis=2).astype(np.uint8)
        return silhouette

    def draw_debug_skeleton(self, kp_3d: ndarray, translated_kp_3d: ndarray, joints_3d: ndarray):
        # mesh1 = pv.PolyData(kp_3d, self.vibe_skeleton_connectivity)
        mesh2 = pv.PolyData(translated_kp_3d, self.vibe_skeleton_connectivity)
        mesh3 = pv.PolyData(joints_3d, self.vibe_skeleton_connectivity)
        return self._plot(# mesh1, Color.red, Color.blue,
                          mesh2, Color.yellow, Color.green,
                          mesh3, Color.cyan, Color.magenta,
                          is_skeleton=True,
                          title='red-blue=vibe, yellow-green=vibe translated, cyan-magenta=model joints'
                          )

    @staticmethod
    def to_pyvista_polygon(polygons: ndarray):
        """Adds the number of vertices in each polygon to the start of the polygon vertex list."""
        if not isinstance(polygons, ndarray):
            raise ValueError
        if len(polygons.shape) != 2:
            raise ValueError
        n_polygons, n_vertices = polygons.shape
        prefix_array = np.full((n_polygons, 1), n_vertices)
        return np.concatenate((prefix_array, polygons), axis=1)

    def set_vtk_camera_pose(self, camera_pose):
        if self.camera is not None:
            t_camera_to_world = Camera.vtk_to_matrix_camera_pose(camera_pose)
            self.camera = Camera(t_camera_to_world=t_camera_to_world,
                                 image_h=self.camera.image_h,
                                 image_w=self.camera.image_w,
                                 fx=self.camera.fx, fy=self.camera.fy)
        else:
            self.vtk_camera = camera_pose


class DebugRenderer:
    def __init__(self, image_width: int, image_height: int, model: Model, joint_indices: ndarray):
        self.image_width = image_width
        self.image_height = image_height
        self.renderer = Renderer(RenderMode.blocking, model=model, show_axes=True, joint_indices=joint_indices)

    def debug_2d(self, p2d, kp_2d=None):
        print('2d debug')
        image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        draw_pixels_on_image(image, p2d, Color.blue)
        if kp_2d is not None:
            draw_pixels_on_image(image, kp_2d, Color.red)
        view_image_blocking(image, "projected keypoints, red=vibe, blue=model")

    def debug_3d(self, p3d, kp_3d, kp_3d_translation):
        print("3d debug")
        joints_3d_bbox = {'min': p3d.min(axis=0), 'max': p3d.max(axis=0)}
        kp_3d_bbox = {'min': kp_3d.min(axis=0), 'max': kp_3d.max(axis=0)}
        print(f'translation: {kp_3d_translation}')
        print(f'joint 3d bounding box: {joints_3d_bbox}')
        print(f'kp 3d bounding box: {kp_3d_bbox}')
        camera_pose, _ = self.renderer.draw_debug_skeleton(kp_3d, kp_3d + kp_3d_translation, p3d)
        self.renderer.set_vtk_camera_pose(camera_pose)

    def debug_silhouette(self, contour_pixels: ndarray, image_silhouette: ndarray, idt: ndarray,):
        print('silhouette debug')
        print('max idt entry: ', idt.max())
        image = np.zeros((image_silhouette.shape[0], image_silhouette.shape[1], 3), dtype=np.uint8)
        image[image_silhouette] = Color.red
        draw_pixels_on_image(image, contour_pixels, Color.green)
        # image[model_silhouette] = Color.green
        view_image_blocking(image, 'red=image silhouette, green=model silhouette')
        view_image_blocking(idt, 'idt')

    def bad_silhouette(self):
        print("### SKIPPED SILHOUETTE, NO PIXEL WAS PAINTED ###")
        self.renderer.draw_model()
