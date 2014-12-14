#!/usr/bin/python2

import argparse
from collections import namedtuple

import cv2
import numpy as np
from scipy.spatial import KDTree as kdt
from matplotlib import pyplot as plt

# FeatureDescriptor:
# + pos: np.array of pixel positions in (row, column) coordinates
# + desc: np.array of feature vectors collected at positions given in @pos
FeatureDescriptor = namedtuple('FeatureDescriptor', 'pos desc')

def find_features(img, method='SIFT', nfeatures=1000):
    '''
    Input:
        img: a cv2 image (color images are supported)
        method: may be 'SIFT' or 'MOPS'
        nfeatures: maximum number of features to find
    Output:
        features: a FeatureDescriptor object
    '''

    if method == 'SIFT':
        sift = cv2.SIFT(nfeatures)
        kp, des = sift.detectAndCompute(img, None)
        return FeatureDescriptor([K.pt for K in kp], des)
    elif method == 'MOPS':
        keyPts = ANMS(img, nfeatures) 
        patches, coords = preparePatches(img, keyPts)
        return FeatureDescriptor(coords, patches)

def find_matches(features1, features2, method='SIFT', nmatches=200):
    '''
    Input:
        features1, features2: FeatureDescriptor for each image
        method: may be 'SIFT' or 'MOPS'
        nmatches: maximum number of correspondence points to find
    Output:
        points1: a list of (r, c) points in the first image
        points2: a list of matching (r, c) points in the second image
    '''
    
    if method == 'SIFT':
        points1 = []
        points2 = []
        des1 = features1.desc
        des2 = features2.desc
        flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5},
                                      {'checks': 50})
        matches = flann.knnMatch(des1, des2, k=2)
        matches = sorted(matches, key=lambda (m, n): m.distance / n.distance)
        for m, n in matches:
            if m.distance < 0.8*n.distance:
                points1.append(features1.pos[m.queryIdx])
                points2.append(features2.pos[m.trainIdx])
            if len(points1) >= nmatches:
                break
        return points1, points2
    elif method == 'MOPS':
        patches1, coords1 = features1.desc, features1.pos
        patches2, coords2 = features2.desc, features2.pos
        matched1, matched2 = matchPoints(patches1, patches2, coords1, coords2)
        npts = min(nmatches, min(len(matched1), len(matched2)))
        return matched1[:npts], matched2[:npts]

def find_fundamental_matrix(points1, points2):
    '''
    Input:
        points1: a list of (r, c) points in the first image
        points2: a list of matching (r, c) points in the second image
    Output:
        F: np.matrix, the fundamental matrix of the stereo pair
    '''
    F, mask = cv2.findFundamentalMat(np.array(points1), np.array(points2))

    error = 0.0
    for pt1, pt2 in zip(points1, points2):
        a, b = pt1
        c, d = pt2
        e = np.matrix([c, d, 1]) * (F * np.matrix([[a], [b], [1]]))
        error += abs(e[0, 0])

    print "Fundamental matrix:\n", F
    print "-> Approximation error =", error
    print "-> # of correspondence points =", len(points1)
    return F

def find_epipoles(F):
    '''
    Input:
        F: Fundamental matrix
    Output:
        o: 3x1 np.matrix, epipole in image1
        op: 3x1 np.matrix, epipole in image2
        (See http://www.robots.ox.ac.uk/~vgg/hzbook/hzbook2/HZepipolar.pdf)
    '''
    U, S, V = np.linalg.svd(F)
    o = np.matrix(V[-1]).getT()     # Approx. right-null vector.
    op = np.matrix(U[:, -1]).getT() # Approx. left-null vector.

    error = 0.0
    error += np.linalg.norm(F*o)
    error += np.linalg.norm(op.getT() * F)

    print "Epipole 1:\n", o
    print "Epipole 2:\n", op
    print "-> Approximation error =", error
    return o, op
 
######################
## <HELPER METHODS> ##
######################
def blockMax(im, blockSize=3):
  filtered = np.zeros(im.shape).astype("bool")
  #filtered = im.copy()
  for r in xrange(im.shape[0]-blockSize+1):
    for c in xrange(im.shape[1]-blockSize+1):
      block = im[r:r+blockSize, c:c+blockSize]
      filtered[r:r+blockSize, c:c+blockSize] |= block == np.max(block)
  return im*(filtered.astype(float))

def findHarris(im, blockSize=3, ksize=3, k=0.04, edgeWidth=20):
  gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype("float32")
  corners = cv2.cornerHarris(gray, blockSize, ksize, k)
  dst = blockMax(corners, 3)
  dst[:edgeWidth,:] = 0.0
  dst[im.shape[0]-edgeWidth:,:] = 0.0
  dst[:,:edgeWidth] = 0.0
  dst[:,im.shape[1]-edgeWidth:] = 0.0
  return dst

def maxCoords(arr, n):
  topr, topc = np.unravel_index(arr.ravel().argsort()[::-1][:n], arr.shape)
  topCoords = np.zeros(arr.shape)
  topCoords[topr, topc] = arr[topr, topc]
  return topCoords

def preparePatches(im, pts, blockSize=20, sampleRate=5):
  gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
  numr, numc = pts.shape
  r,c = np.nonzero(pts)
  coords = np.array(zip(r,c))
  blurred = cv2.GaussianBlur(gray, (0,0), 1)
  mat = []
  for coord in coords:
    row, col = coord
    top = max(0, row-blockSize)
    bot = min(numr-1, row+blockSize)
    left = max(0, col-blockSize)
    right = min(numc-1, col+blockSize)
    box = blurred[top:bot, left:right]

    sub = box[::sampleRate, ::sampleRate].flatten().astype(float)
    sub = (sub-np.mean(sub))/np.std(sub)
    mat.append(sub)
  return np.array(mat), coords

def ANMS(im, n, crobust=0.9):
  """ Does Adaptive Non-Maximal Supression to
  Obtain N Key Points from Harris Corners
  """
  corners = findHarris(im)
  numr, numc = corners.shape
  radii = np.zeros(corners.shape)
  r,c = np.nonzero(corners)
  coords = np.array(zip(r,c))
  for coord in coords:
    row, col = coord
    for radius in range(1, max(numr, numc)):
      top = max(0, row-radius)
      bot = min(numr-1, row+radius+1)
      left = max(0, col-radius)
      right = min(numc-1, col+radius+1)
      box = corners[top:bot, left:right]
      if corners[row,col] < crobust*(np.max(box)):
        radii[row,col] = radius
        break
  return maxCoords(radii, n)

def matchPoints(patch1, patch2, coord1, coord2, thresh=0.4):
  """ Matches a pair of keypoints from two images,
  based on feature descriptors (image patches)
  """
  tree = kdt(patch2)
  dists, idx = tree.query(patch1, k=2)

  ratios = dists[:,0]/dists[:,1]
  patch2Idx = idx[ratios < thresh][:,0]
  patch1Idx = np.arange(len(patch1))[ratios < thresh]

  matched1 = coord1[patch1Idx]
  matched2 = coord2[patch2Idx]
  return matched1, matched2

#######################
## </HELPER METHODS> ##
#######################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('img1')
    parser.add_argument('img2')
    args = parser.parse_args()

    img1 = cv2.imread(args.img1)
    img2 = cv2.imread(args.img2)  

    ## Test for SIFT
    features1 = find_features(img1)
    features2 = find_features(img2)
    points1, points2 = find_matches(features1, features2)
    F = find_fundamental_matrix(points1, points2)
    o, op = find_epipoles(F)

    ## Test for MOPS
    #mops_feat1 = find_features(img1, method="MOPS")
    #mops_feat2 = find_features(img2, method="MOPS")
    #points1, points2 = find_matches(mops_feat1, mops_feat2, method="MOPS")
    #print zip(points1, points2)
