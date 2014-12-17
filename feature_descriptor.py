#!/usr/bin/python2

import argparse
from collections import namedtuple

import cv2
import numpy as np
import random
from scipy.spatial import KDTree as kdt
from matplotlib import pyplot as plt

import reconstruct

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

def find_matches(features1, features2, method='SIFT', nmatches=220):
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

        # XXX: it's unclear why this helps.
        matches = sorted(matches, key=lambda (m, n): m.distance / n.distance)

        # Applying the ratio test seems to prune many bad matches.
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

def normalize_points(img, points):
    '''
    Input:
        img: cv2 image
        points: array of (row, col) points from img
    Output:
        pointsn: normalized point array
    '''
    h, w = map(float, img.shape[:2])
    return np.array([np.array([r/h, c/w]) for (r, c) in points])

def find_fundamental_matrix(points1, points2):
    '''
    Input:
        points1: a list of (r, c) points in the first image
        points2: a list of matching (r, c) points in the second image
    Output:
        F: np.matrix, the fundamental matrix of the stereo pair
    '''
    bestF = None
    bestPts = None
    mostInliers = np.float("-inf")
    n = len(points1)

    for i in range(4000):
        #Pick random sample of size k = 8, without repetition
        k = 8
        indices = [i for i in range(n)]
        res = None
        #res must have rank 8, otherwise it is degenerate
        while res == None or np.linalg.matrix_rank(res) != 8:
            random_idx = random.sample(indices, k)

            #samples frmo points1 and points2
            sub_points1 = []
            sub_points2 = []
            for idx in random_idx:
                sub_points1.append(points1[idx,:])
                sub_points2.append(points2[idx,:])
            sub_points1 = np.array(sub_points1)
            sub_points2 = np.array(sub_points2)

            #Break each sample down into elements A, B
            A = sub_points1[:,0]
            Ap = sub_points2[:,0]
            B = sub_points1[:,1]
            Bp = sub_points2[:,1]

            #creating res for res*x = b 
            ApA = np.multiply(Ap, A) #shape (k, 1)
            ApB = np.multiply(Ap, B)
            BpA = np.multiply(Bp, A)
            BpB = np.multiply(Bp, B)
            res = np.column_stack((ApA, ApB, Ap, BpA, BpB, Bp, A, B)) #(k, 8)

        #creating b for res*x = b
        b = np.zeros((k,1))
        b[:,0] = -1

        #Construct F using lstsq
        F = np.linalg.lstsq(res, b)[0]
        F = np.append(F, [1])
        F = np.reshape(F, (3,3)) #F is now 3x3 where F_3,3 is 1

        error = 0.0
        count = 0
        pts = []
        for i, (pt1, pt2) in enumerate(zip(points1, points2)):
            a, b = pt1
            c, d = pt2
            e = np.matrix([c, d, 1]) * (F * np.matrix([[a], [b], [1]]))
            err = e[0, 0]**2
            #TODO:  finding a good threshold
            if err < 0.0001:
                error += err
                #Increase number of inliers
                count += 1
                pts.append([pt1, pt2])
        #If there are more inliers, update F
        if count > mostInliers:
            mostInliers = count
            bestF = F
            bestPts = pts

    #Use bestF found, with most inliers
    print "Fundamental matrix:\n", bestF
    return np.matrix(bestF), bestPts
 
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
    parser.add_argument('--MOPS', action='store_true')
    args = parser.parse_args()

    img1 = cv2.imread(args.img1)
    img2 = cv2.imread(args.img2)  

    ## Test for SIFT
    if args.MOPS:
        features1 = find_features(img1, method='MOPS')
        features2 = find_features(img2, method='MOPS')
        points1, points2 = find_matches(features1, features2, method='MOPS')
    else:
        features1 = find_features(img1)
        features2 = find_features(img2)
        points1, points2 = find_matches(features1, features2)

    # XXX: test if normalization actually helps.
    points1, points2 = map(normalize_points, (img1, img2), (points1, points2))

    F, selected_pts = find_fundamental_matrix(points1, points2)

    # XXX: check what the effect is on the reconstruction error.
    # points1, points2 = cv2.correctMatches(F, np.array([points1]),
    #         np.array([points2]))
    # points1 = points1[0]
    # points2 = points2[0]

    o, op = reconstruct.find_epipoles(F)
    P1, P2 = reconstruct.approx_perspective_projections(F)
    points3d = []
    depth_err = 0.0
    #for u1, u2 in zip(points1, points2):
    for u1, u2 in selected_pts:
        X, err = reconstruct.reconstruct(P1, P2, u1, u2)
        # print "Mapping", (u1, u2), "to:\n", X, "(error = %f)" % (err,)
        #if err < reconstruct.ERROR_THRESH:
        points3d.append(X)
        depth_err += err

    print "Total depth reconstruction error:", depth_err
    print "Accepted", len(points3d), "of", len(points1), "correspondences"
    reconstruct.render(points3d)

    # XXX: Why does opencv's triangulation method look like garbage?
    # points4d = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
    # points3d_cv2 = [np.array([x/t, y/t, z/t]) for x, y, z, t in zip(*points4d)]
    # reconstruct.render(points3d_cv2)

    ## Test for MOPS
    #mops_feat1 = find_features(img1, method="MOPS")
    #mops_feat2 = find_features(img2, method="MOPS")
    #points1, points2 = find_matches(mops_feat1, mops_feat2, method="MOPS")
    #print zip(points1, points2)
