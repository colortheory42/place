"""
Raycasting utilities.
Möller–Trumbore algorithm for ray-triangle intersection.
"""

import numpy as np


def ray_intersects_triangle(ray_origin, ray_dir, v0, v1, v2):
    """
    Möller–Trumbore intersection algorithm.
    Returns (distance, triangle) if hit, None otherwise.
    """
    epsilon = 0.0000001

    edge1 = np.array(v1) - np.array(v0)
    edge2 = np.array(v2) - np.array(v0)

    h = np.cross(ray_dir, edge2)
    a = np.dot(edge1, h)

    if -epsilon < a < epsilon:
        return None

    f = 1.0 / a
    s = np.array(ray_origin) - np.array(v0)
    u = f * np.dot(s, h)

    if u < 0.0 or u > 1.0:
        return None

    q = np.cross(s, edge1)
    v = f * np.dot(ray_dir, q)

    if v < 0.0 or u + v > 1.0:
        return None

    t = f * np.dot(edge2, q)

    if t > epsilon:
        return (t, (v0, v1, v2))

    return None
