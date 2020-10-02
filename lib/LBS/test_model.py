import numpy as np
from lib.LBS.Model import Joint, AnimatedModel

x = Joint(name="x", local_bind_transform=np.eye(4), index=0)
y = Joint(name="y", local_bind_transform=np.eye(4), index=1)
z = Joint(name="z", local_bind_transform=np.eye(4), index=2)
w = Joint(name="w", local_bind_transform=np.eye(4), index=3)
u = Joint(name="u", local_bind_transform=np.eye(4), index=4)
y.add_child(z)
y.add_child(w)
y.add_child(u)
x.add_child(y)
x.t_model_to_world = np.eye(4)
y.t_model_to_world = 2*np.eye(4)
z.t_model_to_world = 3*np.eye(4)
w.t_model_to_world = 4*np.eye(4)
u.t_model_to_world = 5*np.eye(4)

animated_model = AnimatedModel(None, None, x, 5)
joints_transforms = animated_model.get_joints_transforms()
print(joints_transforms.shape)
