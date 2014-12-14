#!/usr/bin/python2

import cv2
import numpy as np
import numpy.linalg as LA

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
        [1, 0, 0],
        [0, 0, 1]
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

    pass
