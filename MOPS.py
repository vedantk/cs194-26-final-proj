import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree as kdt
import cv2
import os

desc = '''cs194-26 -- MOPS.py

Two functions to obtain keypoints as well as 
match between key points
Author: Siddhartho Bhattacharya'''

######################
## <HELPER METHODS> ##
######################
def bgr_to_rgb(img):
    if len(img.shape) > 2:
        return img[:, :, ::-1]
    return img

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
#######################
## </HELPER METHODS> ##
#######################

def matchPoints(im1, im2, pts1, pts2, thresh=0.4):
  """ Matches a pair of keypoints from two images,
  based on feature descriptors (image patches)
  """
  patch1, coord1 = preparePatches(im1, pts1)
  patch2, coord2 = preparePatches(im2, pts2)
  tree = kdt(patch2)
  dists, idx = tree.query(patch1, k=2)

  ratios = dists[:,0]/dists[:,1]
  patch2Idx = idx[ratios < thresh][:,0]
  patch1Idx = np.arange(len(patch1))[ratios < thresh]

  matched1 = coord1[patch1Idx]
  matched2 = coord2[patch2Idx]
  return matched1, matched2

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


if __name__ == '__main__':
  '''For testing only!'''
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('im1')
  parser.add_argument('im2')
  args = parser.parse_args()

  im1 = cv2.imread(args.im1)
  im2 = cv2.imread(args.im2)
  rows, cols, c = im1.shape
  npts = 500
  topCoords1 = ANMS(im1, npts)
  topCoords2 = ANMS(im2, npts)

  matched1, matched2 = matchPoints(im1, im2, topCoords1, topCoords2)
  del topCoords1
  del topCoords2

  mask1 = np.zeros((im1.shape[0], im1.shape[1])).astype("float32")
  mask1[matched1[:,0], matched1[:,1]] = 1.0
  mask1 = cv2.dilate(mask1, None)

  mask2 = np.zeros((im2.shape[0], im2.shape[1])).astype("float32")
  mask2[matched2[:,0], matched2[:,1]] = 1.0
  mask2 = cv2.dilate(mask2, None)
  f = plt.figure()
  f.add_subplot(1,2,1)
  plt.imshow(bgr_to_rgb(im1))
  plt.scatter(matched1[:,1], matched1[:,0], c=np.arange(matched1.shape[0]))
  f.add_subplot(1,2,2)
  plt.imshow(bgr_to_rgb(im2))
  plt.scatter(matched2[:,1], matched2[:,0], c=np.arange(matched2.shape[0]))
  plt.show()

