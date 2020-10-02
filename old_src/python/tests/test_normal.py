#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


# Plot:

def plot_img_vectors(img, x_gradient, y_gradient, title):
    plot_binary_img(img, title)
    for i in range(len(x_gradient)):
        if i % 10 != 0:
            continue
        for j in range(len(x_gradient)):
            if j % 10 != 0:
                continue
            origin = i, j
            plt.quiver(*origin, x_gradient[i][j],
                       y_gradient[i][j], color=['r'], scale=21)
    plt.show()


def plot_binary_img(img, title):
    plt.title(title)
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j] != 0:
                plt.scatter(i, j)


def plot_vectors(origins, vectors, title):
    plt.title(title)
    for i in range(len(vectors)):
        origin = origins[i][0], origins[i][1]
        plt.scatter(origins[i][0], origins[i][1])
        plt.quiver(*origin, vectors[i][0],
                   vectors[i][1], color=['r'], scale=21)
    plt.show()


# Create:

def create_diamond_img(radius):
    N = 2 * radius
    H = abs(np.arange(1-N, N+1, 2, dtype=float))//2
    V = (N-1)//2-H[:, None]
    diamond = (H == V)*1
    return diamond


def create_circle_img():
    # xx and yy are 200x200 tables containing the x and y coordinates as values
    # mgrid is a mesh creation helper
    xx, yy = np.mgrid[:200, :200]
    # circles contains the squared distance to the (100, 100) point
    # we are just using the circle equation learnt at school
    circle = (xx - 100) ** 2 + (yy - 100) ** 2
    # donuts contains 1's and 0's organized in a donut shape
    # you apply 2 thresholds on circle to define the shape
    donut = np.logical_and(circle < (6400 + 60), circle > (6400 - 60))
    return donut


# Test:

def test_curve():
    arr = np.array([[0.,   0.], [0.3,   0.], [1.25,  -0.1], [2.1,  -0.9],
                    [2.85,  -2.3], [3.8,  -3.95], [5.,  -5.75], [6.4,  -7.8],
                    [8.05,  -9.9], [9.9, -11.6], [12.05, -12.85], [14.25, -13.7],
                    [16.5, -13.8], [19.25, -13.35], [21.3, -12.2], [22.8, -10.5],
                    [23.55,  -8.15], [22.95,  -6.1], [21.35,  -3.95], [19.1,  -1.9]])
    # arr = np.array([[0. , 1.], [1. , 1.], [2. , 1.]])
    plt.scatter(arr[:, 0], arr[:, 1])
    plt.title('Points')
    plt.show()

    normal = calc_normal(arr)
    plot_vectors(arr, normal, 'Normal')


def test_diamond():
    diamond = create_diamond_img(5)
    plot_binary_img(diamond, 'diamond')
    plt.show()

    diamond_grad = np.gradient(diamond)
    plot_img_vectors(
        diamond, diamond_grad[0], diamond_grad[1], 'diamond gradients')


def test_circle():
    circle = create_circle_img()
    plot_binary_img(circle, 'circle')
    plt.show()

    circle = circle.astype(int)
    circle_grad = np.gradient(circle)
    plot_img_vectors(circle, circle_grad[0],
                     circle_grad[1], 'circle_gradients')


def main():
    try:
        # test_curve()
        test_circle()
    except Exception as e:
        print(e)
        pass


if __name__ == '__main__':
    main()
