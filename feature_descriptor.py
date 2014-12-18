#!/usr/bin/python2

import argparse
from collections import namedtuple
import random
import math

import cv2
import numpy as np
import numpy.linalg as LA
from scipy.spatial import KDTree as kdt
from matplotlib import pyplot as plt
import os

import reconstruct

# FeatureDescriptor:
# + pos: np.array of pixel positions in (row, column) coordinates
# + desc: np.array of feature vectors collected at positions given in @pos
FeatureDescriptor = namedtuple('FeatureDescriptor', 'pos desc')

def find_features(img, method='SIFT', nfeatures=250):
    '''
    Input:
        img: a cv2 image (color images are supported)
        method: may be 'SIFT' or 'MOPS'
        nfeatures: maximum number of features to find
    Output:
        features: a FeatureDescriptor object
    '''

    if method == 'SIFT':
        sift = cv2.SIFT(nfeatures=nfeatures, edgeThreshold=10)
        kp, des = sift.detectAndCompute(img, None)
        return FeatureDescriptor([K.pt for K in kp], des)
    elif method == 'MOPS':
        keyPts = ANMS(img, nfeatures) 
        patches, coords = preparePatches(img, keyPts)
        return FeatureDescriptor(coords, patches)

def find_matches(features1, features2, method='SIFT', nmatches=225):
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
        F_err: approximation error
        outliers: set of outlier indices
    '''

    assert len(points1) == len(points2)
    candidates = range(len(points1))

    F = None
    F_err = float('inf')
    ninliers = 0
    outliers = []
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
                F = cur_F
        else:
            # We now have at least 50% of the points as inliers.
            # In this regime, work to just decrease the inlier error.
            if F_err > in_error and cur_ninliers >= 0.5*len(points1):
                print ".",
                F_err, ninliers, outliers = in_error, cur_ninliers, cur_outliers
                F = cur_F

    print "Fundamental matrix:\n", F
    print "-> Approximation error =", F_err, "(given", len(points1), "matches)"
    print "-> Found", ninliers, "inliers"
    return F, F_err, set(outliers)
 
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
    #os.system("./powercrust -m 100000 -i 3d.pts")
    #povray("pc.off")

def povray(off_file):
    '''
    Input:
        f:  a .off file with faces of image
    Parse pts into appropriate format and pass
    to povray.
    '''
    #OFF
    #number of pts, number of faces, number of lines
    #3d pts 
    #faces
    all_points = []
    with open(off_file, "r") as inp:
        lines = inp.readlines()
    num_pts, num_faces, num_lines = lines[1].split()
    num_pts, num_faces, num_lines = int(num_pts), int(num_faces), int(num_lines)
    with open("3dmesh.pov", "w") as out:
        start = 2 + num_pts
        polygons = []
        triangles = []
        for i in range(start, start + num_faces):
            line = lines[i]
            line = line.split()
            num_pts = int(line[0])
            if num_pts > 3:
                output = "polygon { " + str(num_pts) + ","
            else:
                output = "triangle { "
            for i in range(1, num_pts + 1):
                idx = int(line[i])
                pt = lines[idx + 2].split()  #string of three floats separated by space
                all_points.append(map(float, pt))
                #TODO:  STILL causes "Possible Parse Error: Singular matrix in MInvers."
                x = str(math.floor(float(pt[0]) * 10000) / 10000)
                y = str(math.floor(float(pt[1]) * 10000) / 10000)
                z = str(math.floor(float(pt[2]) * 10000) / 10000)
                output += "<" + x + ", " + y + ", " + z + ">"
                if i != num_pts:
                    output += ", "
            if num_pts > 3:
                output += "\n pigment { color rgb <1, 0, 0> }"
                output += " } \n"
                polygons.append(output)
            else:
                output += " } \n"
                triangles.append(output)
        out.write("mesh { \n")
        for t in triangles:
            out.write(t)
        out.write("texture {\npigment { color rgb<0.9, 0.9, 0.9> }\nfinish { ambient 0.2 diffuse 0.7 }\n }\n")
        out.write(" } \n")
        #no mesh
        for p in polygons:
            out.write(p)
        all_points = np.array(all_points)
        maxT = np.max(all_points, axis=0)
        minB = np.min(all_points, axis=0)
        lookAt = minB + 0.5*(maxT-minB)
        k_dist = 5
        camera = lookAt + k_dist*(maxT[0] - minB[0])*np.array([1,0,0])
        out.write("camera {\nlocation <" + str(camera[0]) + ", " + str(camera[1]) + ", " + str(camera[2]) +">\n")
        out.write("look_at <" + str(lookAt[0]) + ", " + str(lookAt[1]) + ", " + str(lookAt[2]) + ">\n}")






#######################
## </HELPER METHODS> ##
#######################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('img1')
    parser.add_argument('img2')
    parser.add_argument('--MOPS', action='store_true')
    args = parser.parse_args()

    img1 = cv2.cvtColor(cv2.imread(args.img1), cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(cv2.imread(args.img2), cv2.COLOR_BGR2GRAY)

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

    F, F_err, outliers = find_fundamental_matrix(points1, points2)

    # XXX: test if correctMatches improves reconstruction error.
    # points1, points2 = cv2.correctMatches(F, np.array([points1]),
    #         np.array([points2]))
    # points1 = points1[0]
    # points2 = points2[0]

    o, op = reconstruct.find_epipoles(F)
    P1, P2 = reconstruct.approx_perspective_projections(F)
    points3d = []
    reconstruct_err = 0.0
    for i, (u1, u2) in enumerate(zip(points1, points2)):
        if i in outliers:
            continue
        X, err = reconstruct.reconstruct(P1, P2, u1, u2)
        # print "Mapping", (u1, u2), "to:\n", X, "(error = %f)" % (err,)
        points3d.append(X)
        reconstruct_err += err
    print "Total depth reconstruction error:", reconstruct_err
    print "Accepted", len(points3d), "of", len(points1), "correspondences"
    reconstruct.render(points3d)

    #Uncomment if you'd like to run powercrust on points3d
    powercrust(points3d)


    ## Test for MOPS
    #mops_feat1 = find_features(img1, method="MOPS")
    #mops_feat2 = find_features(img2, method="MOPS")
    #points1, points2 = find_matches(mops_feat1, mops_feat2, method="MOPS")
    #print zip(points1, points2)
