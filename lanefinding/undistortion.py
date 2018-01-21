import pickle
import matplotlib.image as mpimg
import cv2

images = ["../camera_cal/calibration1.jpg",
          "../camera_cal/calibration2.jpg",
          "../camera_cal/calibration3.jpg",
          "../camera_cal/calibration4.jpg",
          "../camera_cal/calibration5.jpg"]

pickle_calibration = pickle.load(open("../pickle_saved/pickle_calibration.p", "rb"))
objpoints = pickle_calibration["objpoints"]
imgpoints = pickle_calibration["imgpoints"]
for image in images:
    img = mpimg.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imshow(image, dst)
    cv2.waitKey(1000)

cv2.destroyAllWindows()

undistort_mtx_dist = {"mtx":mtx, "dist":dist}
pickle.dump(undistort_mtx_dist, open("../pickle_saved/pickle_un_distortion.p", "wb"))
