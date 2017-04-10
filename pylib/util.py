from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np


def calculate_rgb(shift, radius, theta, psi):
    '''
    In the spherical coordinate, calculate RGB value
    '''
    d2r = np.pi / 180.  # degree to radian
    r = radius * np.cos(theta * d2r) * np.sin(psi * d2r)
    g = radius * np.sin(theta * d2r) * np.sin(psi * d2r)
    b = radius * np.cos(psi * d2r)
    result = np.array([r, g, b]) + shift
    return tuple([round(result[0], 3),
                  round(result[1], 3),
                  round(result[2], 3)])


# define an infinite class color retriever
def retrieve_n_class_color_sphere(N):
    '''
    Retrieve class color, over the sphere object inside RGB 3D cubic box.
    (intend to keep class color strict convex)
    Input: class number
    Output: list of RGB color code
    '''
    color_list = []
    center = np.array([0.5, 0.5, 0.5])
    radius = 0.5

    interval = 90
    np.random.seed(1)  # pre-define the seed for consistency
    while len(color_list) < N:
        the_list = []
        for psi in np.arange(0, 360 + 0.1 * interval, interval):
            for theta in np.arange(0, 360 + 0.1 * interval, interval):
                new_color = calculate_rgb(center, radius, theta, psi)
                if new_color not in color_list:
                    the_list.append(new_color)
        the_list = list(set(the_list))
        np.random.shuffle(the_list)
        color_list.extend(the_list)
        interval = interval / 2.

    return color_list[:N]


def retrieve_n_class_color_cubic(N):
    '''
    retrive color code for N given classes
    Input: class number
    Output: list of RGB color code
    '''

    # manualy encode the top 8 colors
    # the order is intuitive to be used
    color_list = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (0, 1, 1),
        (1, 0, 1),
        (0, 0, 0),
        (1, 1, 1)
    ]

    # if N is larger than 8 iteratively generate more random colors
    np.random.seed(1)  # pre-define the seed for consistency

    interval = 0.5
    while len(color_list) < N:
        the_list = []
        iterator = np.arange(0, 1.0001, interval)
        for i in iterator:
            for j in iterator:
                for k in iterator:
                    if (i, j, k) not in color_list:
                        the_list.append((i, j, k))
        the_list = list(set(the_list))
        np.random.shuffle(the_list)
        color_list.extend(the_list)
        interval = interval / 2.0

    return color_list[:N]
