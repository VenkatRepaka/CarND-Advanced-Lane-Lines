import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob

nx = 9
ny = 6

offset_x = 80
offset_y = 80

undistort_mtx_dist = pickle.load(open("../pickle_saved/pickle_un_distortion.p", "rb"))
mtx = undistort_mtx_dist["mtx"]
dist = undistort_mtx_dist["dist"]

# images = glob.glob('../camera_cal/cal*.jpg')
images = glob.glob("../test_images/*.jpg")
for image in images:
    img = mpimg.imread(image)
    rows, cols, depth = img.shape
    undistort = cv2.undistort(img, mtx, dist, None, mtx)
    gray = cv2.cvtColor(undistort, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    if ret:
        cv2.drawChessboardCorners(undistort, (nx, ny), corners, ret)
        src = np.float32([corners[0], corners[nx - 1], corners[-1], corners[-nx]])
        dst = np.float32([[offset_x, offset_y], [cols - offset_x, offset_y],
                          [cols - offset_x, rows - offset_y],
                          [offset_x, rows - offset_y]])
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(undistort, M, (cols, rows))

cv2.destroyAllWindows()
