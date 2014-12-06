import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

#Grab all jpg images in this directory (currently all sample chess images)
images = glob.glob('*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)

        #Melanie note: corners is output of findChessboardCorners
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (7,6), corners,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)

        #cv2 calibration: creturns camera matrix, distortion coefficients, rotation and translation vectors
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

        #Undistortion (example on left12 image)
        #0.  (preprocessing) Use getOptimalNewCameraMatrix
        #If alpha=1, all pixels are retained with some extra black images.
        #It also returns an image ROI which can be used to crop the result.
        img = cv2.imread('left12.jpg')
        h,  w = img.shape[:2]
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

        #Method 1.  cv2.undistort() 
        #Melanie note:  there is also the remapping method, but the output is the same and this is less hairy
        # undistort
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # crop the image (uses roi)
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imwrite('calibresult.png',dst)

        #Re-projection error:
        mean_error = 0
        for i in xrange(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error

        #Melanie note:  goal is for this to be as close to 0 as possible
        print "total error: ", mean_error/len(objpoints)

cv2.destroyAllWindows()
