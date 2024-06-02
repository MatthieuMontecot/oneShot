import numpy as np


def distance(point_a, point_b):
    return np.sqrt(((point_a - point_b) ** 2).sum())


def norm(v):
    return np.sqrt((v ** 2).sum())


def normalize_v(v):
    return v / np.sqrt((v ** 2).sum())


def get_e_theta(theta, e_theta=None):
    if e_theta is None:
        return np.array([np.cos(theta), np.sin(theta)])
    else:
        e_theta[0] = np.cos(theta)
        e_theta[1] = np.sin(theta)
        return e_theta


def orthogonal_dot(vec, point):
    """a line is defined by ax + by + c = 0, it's a way to calculate the constant c (well, -c)
    vec is a directing vector of the line and point is a point of the line
    vec has to be normalized"""
    return - vec[1] * point[0] + vec[0] * point[1]


def cross_lines(point_a, vec_a, point_b, vec_b):
    """returns the intersection point between 2 lines"""
    const_a = orthogonal_dot(vec_a, point_a)
    const_b = orthogonal_dot(vec_b, point_b)
    x = - (vec_a[0] * const_b - const_a * vec_b[0]) / (- vec_a[1] * vec_b[0] + vec_b[1] * vec_a[0])
    y = - (- vec_a[1] * const_b + const_a * vec_b[1]) / (- vec_a[0] * vec_b[1] + vec_b[0] * vec_a[1])
    return np.array([x, y])


def modify_v(v):
    v += (-0.5 + np.random.rand(v.size).reshape(v.shape))
