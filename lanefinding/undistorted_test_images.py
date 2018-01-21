import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import glob

images = glob.glob("../test_images/*.jpg")

undistort_mtx_dist = pickle.load(open("../pickle_saved/pickle_un_distortion.p", "rb"))
mtx = undistort_mtx_dist["mtx"]
dist = undistort_mtx_dist["dist"]

for image in images:
    img = mpimg.imread(image)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=15)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=15)
    mpimg.imsave(image.replace("test_images", "output_images"), dst)

    plt.waitforbuttonpress()
