#!/usr/bin/python2

import random
import operator

import cv2
import numpy as np
import numpy.linalg as LA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def find_fundamental_matrix(points1, points2):
    '''
    Input:
        points1: a list of (r, c) points in the first image
        points2: a list of matching (r, c) points in the second image
    Output:
        F: np.matrix, the fundamental matrix of the stereo pair
        F_err: approximation error
        outliers: set of outlier indices
    '''

    assert len(points1) == len(points2)
    candidates = range(len(points1))

    F = None
    F_err = float('inf')
    ninliers = 0
    outliers = []
    iters_needed = 0
    max_iters = max(1000, 10*len(points1))

    for j in xrange(max_iters):
        rows = []
        samples = random.sample(candidates, 8)
        for k in samples:
            A, B = points1[k]
            Ap, Bp = points2[k]
            rows.append([Ap*A, Ap*B, Ap, Bp*A, Bp*B, Bp, A, B])
        A = np.matrix(rows)
        b = np.matrix([[-1]] * 8)
        x, residuals, rank, s = LA.lstsq(A, b)

        # The system is under-constrained, no need to waste time evaluating it.
        if rank < 8:
            continue

        cur_F = np.matrix(np.array(list(x) + [[1]]).reshape(3, 3))

        # Estimate # of inliers, inlier error, and outliers.
        in_error = 0.0
        cur_ninliers = 0
        cur_outliers = []
        for i, (pt1, pt2) in enumerate(zip(points1, points2)):
            a, b = pt1
            c, d = pt2
            e = np.matrix([c, d, 1]) * (cur_F * np.matrix([[a], [b], [1]]))
            pt_err = e[0, 0]**2
            if pt_err < 0.01:
                cur_ninliers += 1
                in_error += pt_err
            else:
                cur_outliers.append(i)

        if ninliers < 0.5*len(points1):
            # We don't have very many inliers.
            # In this regime, work to increase the # of inliers.
            if ninliers < cur_ninliers:
                print "o",
                F_err, ninliers, outliers = in_error, cur_ninliers, cur_outliers
                iters_needed = j
                F = cur_F
        else:
            # We now have at least 50% of the points as inliers.
            # In this regime, work to just decrease the inlier error.
            if F_err > in_error and cur_ninliers >= 0.5*len(points1):
                print ".",
                F_err, ninliers, outliers = in_error, cur_ninliers, cur_outliers
                iters_needed = j
                F = cur_F

    print "Fundamental matrix:\n", F
    print "-> Approximation error =", F_err, "(given", len(points1), "matches)"
    print "-> Found", ninliers, "inliers", "after", iters_needed, "iterations"
    return F, F_err, set(outliers)

def find_epipoles(F):
    '''
    Input:
        F: Fundamental matrix (np.matrix)
    Output:
        o: 3x1 np.matrix, epipole in image1
        op: 3x1 np.matrix, epipole in image2
        (See http://www.robots.ox.ac.uk/~vgg/hzbook/hzbook2/HZepipolar.pdf)
    '''

    U, S, VT = LA.svd(F)
    o = np.matrix(VT[-1]).getT()    # Approx. right-null vector.
    op = np.matrix(U[:, -1]).getT() # Approx. left-null vector.

    error = 0.0
    error += LA.norm(F * o)
    error += LA.norm(op * F)

    print "Epipole 1:\n", o
    print "Epipole 2:\n", op
    print "-> Approximation error =", error
    return o, op.getT()

def approx_perspective_projections(F):
    '''
    Input:
        F: Fundamental matrix (np.matrix)
    Output:
        P1, P2: maps from P^3 -> P^2 (3x4 perspective projection matrices)
    '''

    P1 = np.matrix([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ])

    U, (r, s, _), VT = LA.svd(F)
    P2 = np.ndarray((3, 4))
    gamma = s + (r-s)/2             # XXX: check that gamma can be > s.
    E = np.matrix([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1]
    ])
    P2[:3, :3] = U*np.diag([r, s, gamma])*E*VT
    P2[:, 3] = (U*np.matrix([[0], [0], [gamma]])).getT()

    P1, P2 = np.matrix(P1), np.matrix(P2)

    print "Projection matrix 1:\n", P1
    print "Projection matrix 2:\n", P2

    return P1, P2

def reconstruct(P1, P2, u1, u2):
    '''
    Input:
        P1, P2: camera 1 and 2 perspective projection matrices
        u1, u2: corresponding points in (row, column) format
    Output:
        (x, y, z): a point in 3-d space
        error: reconstruction error estimate
    '''

    u, v = u1
    up, vp = u2

    # L: 6x5, x = (x_i, y_i, z_i, w_i, w'_i)^T, b: 6x1.
    L = np.matrix([
        # (P1[0, :] \dot (x_i, y_i, z_i, 1)) - u_iw_i = 0
        [P1[0, 0], P1[0, 1], P1[0, 2], -u, 0], 

        # (P1[1, :] \dot (x_i, y_i, z_i, 1)) - v_iw_i = 0
        [P1[1, 0], P1[1, 1], P1[1, 2], -v, 0], 

        # (P1[2, :] \dot (x_i, y_i, z_i, 1)) - w_i = 0
        [P1[2, 0], P1[2, 1], P1[2, 2], -1, 0], 

        # (P2[0, :] \dot (x_i, y_i, z_i, 1)) - u'_iw'_i = 0
        [P2[0, 0], P2[0, 1], P2[0, 2], 0, -up], 

        # (P2[1, :] \dot (x_i, y_i, z_i, 1)) - v'_iw'_i = 0
        [P2[1, 0], P2[1, 1], P2[1, 2], 0, -vp], 

        # (P2[2, :] \dot (x_i, y_i, z_i, 1)) - w'_i = 0
        [P2[2, 0], P2[2, 1], P2[2, 2], 0, -1], 
    ])

    b = np.matrix([
        [-P1[0, 3]],
        [-P1[1, 3]],
        [-P1[2, 3]],
        [-P2[0, 3]],
        [-P2[1, 3]],
        [-P2[2, 3]],
    ])

    x, residuals, rank, s = LA.lstsq(L, b)
    return x[:3], np.sum([r**2 for r in residuals])

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
