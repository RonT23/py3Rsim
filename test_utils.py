import numpy as np

def create_circle(r, z, T):
    """
    Generate a series of waypoints that form a circle in the XY plane.
    :param T: number of waypoints to create (full circle).
    :param r: circle radius.
    :param z: hight from origin at plane XY.
    :return: a list of [x, y, z] coordinates (in meters).
    """
    dth = 2 * np.pi / T

    P = []
    for i in range(T):
        th = i * dth
        x = r * np.sin(th)
        y = r * np.cos(th)
        P.append([x, y, z])
    return P

def create_letter_R():
    return [
        (0.1, 0.6, 0.0),
        (0.1, 0.6, 0.3),
        (0.1, 0.6, 0.5),
        (0.1, 0.6, 0.7),
        (0.1, 0.6, 0.8),
        (0.2, 0.6, 0.8),
        (0.3, 0.6, 0.8),
        (0.4, 0.6, 0.6),
        (0.3, 0.6, 0.4),
        (0.2, 0.6, 0.4),
        (0.3, 0.6, 0.2),
        (0.4, 0.6, 0.0)
    ]