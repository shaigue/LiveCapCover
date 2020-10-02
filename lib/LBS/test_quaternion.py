from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import numpy as np
# from pyquaternion import Quaternion


def arrays_equal_within_error(array1, array2):
    print(array1)
    print(array2)
    if np.allclose(array1, array2, atol=0.0001):
        print('arrays are equal within small margin error')
    else:
        print('arrays are NOT equal!')


def test1():
    rot1 = R.from_matrix(np.eye(3))
    rot2 = R.from_matrix(np.eye(3))
    rotations = R((rot1, rot2))
    # rotations = R.from_matrix((rot1, rot2))

    key_rots = rotations
    key_times = [0, 1]

    slerp = Slerp(key_times, key_rots)
    times = [0.01, 0.5, 0.99]
    interp_rots = slerp(times)

    print(interp_rots.as_matrix()[0])


def test2():
    return


def main():
    test1()
    # test2()


if __name__ == '__main__':
    main()
