import numpy as np
from Model import Joint, AnimatedModel

x = Joint(index=0, name="x", local_bind_transform=np.eye(4))
y = Joint(index=1, name="y", local_bind_transform=np.eye(4))
z = Joint(index=2, name="z", local_bind_transform=np.eye(4))
w = Joint(index=3, name="w", local_bind_transform=np.eye(4))
u = Joint(index=4, name="u", local_bind_transform=np.eye(4))
y.add_child(z)
y.add_child(w)
y.add_child(u)
x.add_child(y)
x.joint_transform = np.eye(4)
y.joint_transform = 2*np.eye(4)
z.joint_transform = 3*np.eye(4)
w.joint_transform = 4*np.eye(4)
u.joint_transform = 5*np.eye(4)

animated_model = AnimatedModel(None, None, x, 5)
joints_transforms = animated_model.get_joints_transforms()
print(joints_transforms.shape)
