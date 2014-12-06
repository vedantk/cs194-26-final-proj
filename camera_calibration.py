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

images = glob.glob('*.jpg')
print "hi"
print images

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

        #Melanie note: changed corners2 to corners- supposed to be output of findChessboardCorners
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
        # undistort
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # crop the image (uses roi)
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imwrite('calibresult.png',dst)

        #Method 2.  remapping
        # undistort
        #mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
        #dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

        # crop the image
        #x,y,w,h = roi
        #dst = dst[y:y+h, x:x+w]
        #cv2.imwrite('calibresult.png',dst)

        #Melanie:  looks like they both get the same result- method 1 looks easier to code
        #Re-projection error:
        mean_error = 0
        for i in xrange(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            #Melanie:  s/tot_error/mean_error --> should be close to 0
            mean_error += error

        print "total error: ", mean_error/len(objpoints)



cv2.destroyAllWindows()
