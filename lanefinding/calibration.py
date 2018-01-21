import numpy as np
import glob
import cv2
import matplotlib.image as mpimg
import pickle

nx = 9
ny = 6

objp = np.zeros([nx*ny, 3], np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

objpoints = []
imgpoints = []

images = glob.glob('../camera_cal/cal*.jpg')
for image in images:
    img = mpimg.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        cv2.imshow(image, img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

pickle_calibration = {"objpoints": objpoints, "imgpoints": imgpoints}
pickle.dump(pickle_calibration, open("../pickle_saved/pickle_calibration.p", "wb"))

