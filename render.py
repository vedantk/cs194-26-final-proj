#!/usr/bin/python2

import os
import operator

from matplotlib import pyplot as plt

def render_points(points):
    '''
    Input:
        points: array of (x, y, z) locations
    '''

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = map(operator.itemgetter(0), points)
    ys = map(operator.itemgetter(1), points)
    zs = map(operator.itemgetter(2), points)
    ax.scatter(xs, ys, zs, c='r', marker='o')
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.show()

def powercrust(points3d):
    '''
    Input:
        points3d: matrix of (x, y, z) points
    Parse points3d into appropriate format and pass
    to powercrust.  Use powercrust output to visualize mesh
    in geomview.
    '''
    with open("3d.pts", "w") as f:
        for pt in points3d:
            x, y, z = pt
            p = [str(float(x)), str(float(y)), str(float(z)), "\n"]
            f.write(" ".join(p))
    os.system("./powercrust -m 100000 -i 3d.pts && geomview pc.off")
