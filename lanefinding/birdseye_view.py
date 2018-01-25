import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import glob


def birds_eye_view(img, image):
    rows, cols, depth = img.shape
    h, w = img.shape[:2]
    src = np.float32([[490, 482], [810, 482], [1250, 720], [40, 720]])
    # src = np.float32([[530, 470], [760, 470], [1250, 720], [40, 720]])
    dst = np.float32([[0, 0], [1280, 0], [1250, 720], [40, 720]])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (cols, rows), flags=cv2.INTER_LINEAR)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(warped)
    ax2.set_title('Warped Image', fontsize=30)
    plt.waitforbuttonpress()

    mpimg.imsave(image.replace("output_images", "birds_eye_view_images"), warped)


# img = mpimg.imread("../output_images/straight_lines1.jpg")
# plt.imshow(img)
# plt.waitforbuttonpress()
images = glob.glob("../output_images/*.jpg")
for image in images:
    img = mpimg.imread(image)
    birds_eye_view(img, image)
