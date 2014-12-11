#!/usr/bin/python2

from collections import namedtuple

# FeatureDescriptor:
# + pos: a pixel position in (row, column) coordinates
# + desc: a feature vector
FeatureDescriptor = namedtuple('FeatureDescriptor', 'pos desc')

def find_features(img, method='SIFT', nfeatures=1000):
    '''
    Input:
        img: a cv2 image (color images are supported)
        method: may be 'SIFT' or 'MOPS'
    Output:
        features: a list of FeatureDescriptor objects
    '''

    # Convert the image to grayscale if necessary.
    im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) > 2 else img

    if method == 'SIFT':
        sift = cv2.SIFT(nfeatures)
        kp, des = sift.detectAndCompute(img, None)

    elif method == 'MOPS':
        pass

def find_matches(features1, features2, method='SIFT', nmatches=1000):
    '''
    Input:
        features1, features2: list of FeatureDescriptor
        method: may be 'SIFT' or 'MOPS'
        nmatches: maximum number of correspondence points to find
    Output:
        correspondences: a list of [(pt1, pt2), ...]
    '''
    pass
