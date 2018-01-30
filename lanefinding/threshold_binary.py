import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return sbinary


def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobel = np.sqrt(sobelx**2 + sobely**2)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    magbinary = np.zeros_like(scaled_sobel)
    magbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

    return magbinary


def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    direction = np.arctan2(abs_sobely, abs_sobelx)

    dirbinary = np.zeros_like(direction)
    dirbinary[(direction >= thresh[0]) & (direction <= thresh[1])] = 1

    return dirbinary


def select_yellow(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower = np.array([20, 60, 60])
    upper = np.array([38, 174, 250])
    mask = cv2.inRange(hsv, lower, upper)
    return mask


def select_white(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # lower = np.array([202, 202, 202])
    # upper = np.array([255, 255, 255])
    lower = np.array([20, 0, 180])
    upper = np.array([255, 80, 255])
    mask = cv2.inRange(hsv, lower, upper)
    return mask


def comb_thresh(image):
    yellow = select_yellow(image)
    white = select_white(image)
    combined_binary = np.zeros_like(yellow)
    combined_binary[(yellow >= 1) | (white >= 1)] = 1
    return combined_binary


def pipeline(image, img):
    ksize = 7
    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]

    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(30, 180))
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(30, 180))
    mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(80, 150))
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(0.7, 1.3))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    # f.tight_layout()
    # ax1.imshow(img)
    # ax1.set_title('Original Image', fontsize=50)
    # ax2.imshow(combined, cmap='gray')
    # ax2.set_title('Combined', fontsize=50)
    # plt.waitforbuttonpress()

    # Threshold color channel
    s_thresh_min = 170
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    # f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
    # f.tight_layout()
    # ax1.imshow(img)
    # ax1.set_title('Original Image', fontsize=50)
    # ax2.imshow(combined)
    # ax2.set_title('Combined Image', fontsize=50)
    # ax3.imshow(s_binary, cmap='gray')
    # ax3.set_title('S binary', fontsize=50)
    # plt.waitforbuttonpress()

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    # color_binary = np.dstack((np.zeros_like(s_binary), combined, s_binary)) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(s_binary)
    # combined_binary[(s_binary == 1) | (combined == 1)] = 1
    # combined_binary[(s_binary == 1) | (combined == 1)] = 1

    wny_threshold = comb_thresh(img)
    # plt.imshow(wny_threshold)
    # plt.waitforbuttonpress()

    combined_binary[(wny_threshold == 1) | (combined == 1)] = 1

    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(combined, cmap='gray')
    ax2.set_title('Sobel Threshold Image', fontsize=30)
    ax3.imshow(s_binary, cmap='gray')
    ax3.set_title('S binary', fontsize=30)
    ax4.imshow(combined_binary, cmap='gray')
    ax4.set_title('Combined binary', fontsize=30)
    plt.waitforbuttonpress()

    mpimg.imsave(image.replace("birds_eye_view_images", "threshold_images"), combined_binary, cmap="gray")

    return combined_binary


images = glob.glob("../birds_eye_view_images/*.jpg")
for image in images:
    pipeline(image, mpimg.imread(image))

