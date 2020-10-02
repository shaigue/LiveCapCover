# model_path = 'data/models/basic/cow.obj'

# import trimesh
# import pyrender
# trimesh = trimesh.load(model_path)

# mesh = pyrender.Mesh.from_trimesh(trimesh)
# scene = pyrender.Scene()
# scene.add(mesh)
# pyrender.Viewer(scene)


import trimesh
import pyrender
import numpy as np
import math
import collada


red = [1.0, 0.0, 0.0]
green = [0.0, 1.0, 0.0]
blue = [0.0, 0.0, 1.0]
black = [0.0, 0.0, 0.0]
gray = [0.5, 0.5, 0.5]
white = [1.0, 1.0, 1.0]

point_light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=2.0)
spot_light = pyrender.SpotLight(color=[1.0, 1.0, 1.0], intensity=2.0,
                                innerConeAngle=0.05, outerConeAngle=0.5)
directional_light = pyrender.DirectionalLight(
    color=[1.0, 1.0, 1.0], intensity=2.0)

perspective_camera = pyrender.PerspectiveCamera(
    yfov=np.pi / 3.0, aspectRatio=1.414)
ortho_camera = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0)


def display(mesh, light=None, camera=None):
    scene = pyrender.Scene(ambient_light=white)
    scene.add(mesh)
    if light:
        scene.add(light)
    if camera:
        scene.add(camera)
    pyrender.Viewer(scene)


def test1():
    model_path = 'data/models/basic/cow.obj'

    # load the cow model
    tm = trimesh.load(model_path)
    tm.visual.vertex_colors = np.random.uniform(
        size=tm.visual.vertex_colors.shape)
    tm.visual.face_colors = np.random.uniform(size=tm.visual.face_colors.shape)
    m = pyrender.Mesh.from_trimesh(tm, smooth=False)

    # display(m, point_light)
    # display(m, spot_light)
    display(m, directional_light, perspective_camera)
    # display(m, directional_light, ortho_camera)

    # from_points
    pts = tm.vertices.copy()
    colors = np.random.uniform(size=pts.shape)
    m = pyrender.Mesh.from_points(pts, colors=colors)
    display(m, point_light)

    # now we will create a UV sphere trimesh, and then we will instance it len(pts) times!
    # create a UV sphere trimesh
    sm = trimesh.creation.uv_sphere(radius=0.1)
    color = trimesh.visual.to_rgba(red)
    sm.visual.vertex_colors = color
    # sm.visual.vertex_colors = np.random.uniform(size=sm.visual.vertex_colors.shape)

    # homogenous transformation matrics
    tfs = np.tile(np.eye(4), (len(pts), 1, 1))
    tfs[:, :3, 3] = pts

    # givin tfs as poses is called instancing, because we're taking one trimesh - sm - and duplicating it len(tfs) times
    m = pyrender.Mesh.from_trimesh(sm, poses=tfs)
    display(m, directional_light)


def test2():
    model_path = 'data/models/basic/cow.obj'

    # load the cow model
    tm = trimesh.load(model_path)
    tm.visual.vertex_colors = np.random.uniform(
        size=tm.visual.vertex_colors.shape)
    tm.visual.face_colors = np.random.uniform(size=tm.visual.face_colors.shape)
    mesh = pyrender.Mesh.from_trimesh(tm, smooth=False)

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
    nm = pyrender.Node(mesh=mesh, matrix=np.eye(4))
    nl = pyrender.Node(light=light, matrix=np.eye(4))
    nc = pyrender.Node(camera=cam, matrix=np.eye(4))
    scene = pyrender.Scene(
        ambient_light=[1.0, 1.0, 1.0], bg_color=gray)
    scene.add_node(nm)
    scene.add_node(nl, parent_node=nm)
    scene.add_node(nc, parent_node=nm)
    pyrender.Viewer(scene, use_raymond_lighting=True)


def rotation_matrix(theta_x=0, theta_y=0, theta_z=0):
    theta_x = np.deg2rad(theta_x)
    theta_y = np.deg2rad(theta_y)
    theta_z = np.deg2rad(theta_z)
    cos_x = math.cos(theta_x)
    sin_x = math.sin(theta_x)
    cos_y = math.cos(theta_y)
    sin_y = math.sin(theta_y)
    cos_z = math.cos(theta_z)
    sin_z = math.sin(theta_z)
    M_x = np.array([[1, 0, 0, 0],
                    [0, cos_x, -sin_x, 0],
                    [0, sin_x, cos_x, 0],
                    [0, 0, 0, 1]])
    M_y = np.array([[cos_y, 0, sin_y, 0],
                    [0, 1, 0, 0],
                    [-sin_y, 0, cos_y, 0],
                    [0, 0, 0, 1]])
    M_z = np.array([[cos_z, -sin_z, 0, 0],
                    [sin_z, cos_z, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    M = np.dot(M_x, np.dot(M_y, M_z))
    return M


def test3():
    model_path = 'data/models/basic/cow.obj'
    scene = pyrender.Scene(ambient_light=white)

    # load the cow model
    tm = trimesh.load(model_path)
    tm.visual.vertex_colors = np.random.uniform(
        size=tm.visual.vertex_colors.shape)
    tm.visual.face_colors = np.random.uniform(size=tm.visual.face_colors.shape)

    mesh = pyrender.Mesh.from_trimesh(tm, smooth=False)
    pose = np.eye(4)
    x = 5
    dist = [x, x, x]
    pose[:3, 3] = dist
    scene.add(mesh, pose=pose)

    M = rotation_matrix(theta_x=90, theta_y=90)
    tm.apply_transform(M)
    mesh2 = pyrender.Mesh.from_trimesh(tm, smooth=False)
    pose2 = np.eye(4)
    pose2[:3, 3] = [-x for x in dist]
    scene.add(mesh2, pose=pose2)

    pyrender.Viewer(scene)


def test4():
    model_path = 'data/mokmnmjmlngina.daesdf'
    model_path = 'data/models/regina/regina.dae'
    model_path = 'data/models/hip_hop/hip_hop.dae'
    tm = trimesh.exchange.dae.load_collada(model_path)
    # tm = load_collada(model_path)
    tm = trimesh.Trimesh(tm)
    tm = trimesh.load_mesh(tm)
    mesh = pyrender.Mesh.from_trimesh(tm)
    display(mesh)


def test5():
    model_path = 'data/models/duck/duck_triangles.dae'
    mesh = collada.Collada(model_path)
    tm = trimesh.exchange.dae.load_collada(model_path)
    tm = trimesh.Trimesh(tm)
    return


def main():
    test1()
    # test2()
    # test3()
    # test4()
    # test5()


def load_collada(file_obj, resolver=None, **kwargs):
    """
    Load a COLLADA (.dae) file into a list of trimesh kwargs.
    Parameters
    ----------
    file_obj : file object
      Containing a COLLADA file
    resolver : trimesh.visual.Resolver or None
      For loading referenced files, like texture images
    kwargs : **
      Passed to trimesh.Trimesh.__init__
    Returns
    -------
    loaded : list of dict
      kwargs for Trimesh constructor
    """
    # load scene using pycollada
    c = collada.Collada(file_obj)

    # Create material map from Material ID to trimesh material
    material_map = {}
    for m in c.materials:
        effect = m.effect
        # material_map[m.id] = _parse_material(effect, resolver)

    # name : kwargs
    meshes = {}
    # list of dict
    graph = []

    for node in c.scene.nodes:
        _parse_node(node=node,
                    parent_matrix=np.eye(4),
                    material_map=material_map,
                    meshes=meshes,
                    graph=graph,
                    resolver=resolver)

    # create kwargs for load_kwargs
    result = {'class': 'Scene',
              'graph': graph,
              'geometry': meshes}

    return result


if __name__ == '__main__':
    main()
