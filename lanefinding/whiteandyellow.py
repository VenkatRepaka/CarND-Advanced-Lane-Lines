import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


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
    upper = np.array([255,  80, 255])
    mask = cv2.inRange(hsv, lower, upper)
    return mask


def comb_thresh(image):
    yellow = select_yellow(image)
    white = select_white(image)
    combined_binary = np.zeros_like(yellow)
    combined_binary[(yellow >= 1) | (white >= 1)] = 1
    return combined_binary


img = mpimg.imread('../birds_eye_view_images/straight_lines1.jpg')
wny_threshold = comb_thresh(img)
plt.imshow(wny_threshold, cmap='gray')
plt.waitforbuttonpress()
