import cv2
import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from collections import deque


def undistort_image(original_image, mtx, dist):
    undist_img = cv2.undistort(original_image, mtx, dist, None, mtx)
    return undist_img


def warp_image(img, src, dst, img_size):
    # Apply perspective transform
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    Minv = cv2.getPerspectiveTransform(dst, src)

    return warped, M, Minv


def select_yellow(image_for_yellow):
    hsv = cv2.cvtColor(image_for_yellow, cv2.COLOR_RGB2HSV)
    lower = np.array([20, 60, 60])
    upper = np.array([38, 174, 250])
    mask = cv2.inRange(hsv, lower, upper)
    return mask


def select_white(image_for_white):
    image_for_white = cv2.cvtColor(image_for_white, cv2.COLOR_RGB2BGR)
    lower = np.array([202, 202, 202])
    upper = np.array([255, 255, 255])
    mask = cv2.inRange(image_for_white, lower, upper)
    return mask


def comb_thresh(image_to_combine):
    yellow = select_yellow(image_to_combine)
    white = select_white(image_to_combine)
    combined_binary = np.zeros_like(yellow)
    combined_binary[(yellow >= 1) | (white >= 1)] = 1
    return combined_binary


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    if orient == 'x':
        img_s = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    else:
        img_s = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    img_abs = np.absolute(img_s)
    img_sobel = np.uint8(255 * img_abs / np.max(img_abs))

    binary_output = 0 * img_sobel
    binary_output[(img_sobel >= thresh[0]) & (img_sobel <= thresh[1])] = 1
    return binary_output


def color_sobel_combined(warped):
    image_HLS = cv2.cvtColor(warped, cv2.COLOR_RGB2HLS)

    img_L = image_HLS[:, :, 1]
    img_abs_x = abs_sobel_thresh(img_L, 'x', 5, (50, 225))
    img_abs_y = abs_sobel_thresh(img_L, 'y', 5, (50, 225))
    wraped_L = np.copy(cv2.bitwise_or(img_abs_x, img_abs_y))

    img_S = image_HLS[:, :, 2]
    img_abs_x = abs_sobel_thresh(img_S, 'x', 5, (50, 255))
    img_abs_y = abs_sobel_thresh(img_S, 'y', 5, (50, 255))
    wraped_S = np.copy(cv2.bitwise_or(img_abs_x, img_abs_y))

    image_cmb = cv2.bitwise_or(wraped_L, wraped_S)
    image_cmb = gaussian_blur(image_cmb, 25)

    yellow_white_combined = comb_thresh(warped)

    image_cmb_color = np.zeros_like(image_cmb)
    image_cmb_color[(yellow_white_combined >= .5) | (image_cmb >= .5)] = 1

    return yellow_white_combined, image_cmb, image_cmb_color


def gaussian_blur(img, kernel=5):
    # Function to smooth image
    blur = cv2.GaussianBlur(img, (kernel,kernel), 0)
    return blur


def lines_from_histogram(img):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    # quarter_point = np.int(midpoint // 2)
    # Previously the left/right base was the max of the left/right half of the histogram
    # this changes it so that only a quarter of the histogram (directly to the left/right) is considered
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # print('base pts:', leftx_base, rightx_base)

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(img.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 80
    # Set minimum number of pixels found to recenter window
    minpix = 40
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Rectangle data for visualization
    rectangle_data = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        rectangle_data.append((win_y_low, win_y_high, win_xleft_low, win_xleft_high, win_xright_low, win_xright_high))
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                    nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                    nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit, right_fit = (None, None)
    # Fit a second order polynomial to each
    if len(leftx) != 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit = np.polyfit(righty, rightx, 2)

    visualization_data = (rectangle_data, histogram)

    return left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data


def average_recent_fits(binary_warped, left_fit_prev, right_fit_prev):
    recent_left_xfitted.append(left_fit_prev)
    recent_right_xfitted.append(right_fit_prev)

    left_fit_avg = np.mean(recent_left_xfitted, 0)
    right_fit_avg = np.mean(recent_right_xfitted, 0)

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 80
    left_lane_inds = ((nonzerox > (
            left_fit_avg[0] * (nonzeroy ** 2) + left_fit_avg[1] * nonzeroy + left_fit_avg[2] - margin)) &
                      (nonzerox < (left_fit_avg[0] * (nonzeroy ** 2) + left_fit_avg[1] * nonzeroy + left_fit_avg[2]
                                   + margin)))
    right_lane_inds = ((nonzerox > (
            right_fit_avg[0] * (nonzeroy ** 2) + right_fit_avg[1] * nonzeroy + right_fit_avg[2] - margin)) &
                       (nonzerox < (right_fit_avg[0] * (nonzeroy ** 2) + right_fit_avg[1] * nonzeroy + right_fit_avg[2]
                                    + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit_new, right_fit_new = (None, None)
    if len(leftx) != 0:
        # Fit a second order polynomial to each
        left_fit_new = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit_new = np.polyfit(righty, rightx, 2)
    return left_fit_new, right_fit_new, left_lane_inds, right_lane_inds

    return left_fit_new, right_fit_new, left_lane_inds, right_lane_inds


def draw_lines(binary_image, left_fit, right_fit):
    margin = 80
    # left_fit2, right_fit2, left_lane_inds2, right_lane_inds2 = polyfit_using_prev_fit(binary_image, left_fit,
    #                                                                                   right_fit)
    left_fit2, right_fit2, left_lane_inds2, right_lane_inds2 = average_recent_fits(binary_image, left_fit, right_fit)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_image.shape[0] - 1, binary_image.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    left_fitx2 = left_fit2[0] * ploty ** 2 + left_fit2[1] * ploty + left_fit2[2]
    right_fitx2 = right_fit2[0] * ploty ** 2 + right_fit2[1] * ploty + right_fit2[2]

    # Create an image to draw on and an image to show the selection window
    out_img = np.uint8(np.dstack((binary_image, binary_image, binary_image)) * 255)
    window_img = np.zeros_like(out_img)

    # Color in left and right line pixels
    nonzero = binary_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    out_img[nonzeroy[left_lane_inds2], nonzerox[left_lane_inds2]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds2], nonzerox[right_lane_inds2]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area (OLD FIT)
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    return result


def draw_lane(original_img, binary_img, l_fit, r_fit, Minv):
    new_img = np.copy(original_img)
    if l_fit is None or r_fit is None:
        return original_img
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    h, w = binary_img.shape
    ploty = np.linspace(0, h - 1, num=h)  # to cover same y-range as image
    left_fitx = l_fit[0] * ploty ** 2 + l_fit[1] * ploty + l_fit[2]
    right_fitx = r_fit[0] * ploty ** 2 + r_fit[1] * ploty + r_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    # cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255, 0, 255), thickness=15)
    # cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0, 255, 255), thickness=15)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (w, h))
    # Combine the result with the original image
    result = cv2.addWeighted(new_img, 1, newwarp, 0.5, 0)
    return result


def radius_of_curvature_and_deviation(binary_image, left_fit, right_fit, left_lane_indices, right_lane_indices):
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 900  # meters per pixel in x dimension
    h, w = binary_image.shape
    ploty = np.linspace(0, h-1, num=h)
    y_eval = np.max(ploty)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx = nonzerox[left_lane_indices]
    lefty = nonzeroy[left_lane_indices]
    rightx = nonzerox[right_lane_indices]
    righty = nonzeroy[right_lane_indices]

    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

    left_radius = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
    right_radius = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

    car_centre = w/2.0
    left_bottom = left_fit[0]*h**2+left_fit[1]*h+left_fit[2]
    right_bottom = right_fit[0]*h**2+right_fit[1]*h+right_fit[2]
    lane_centre = (left_bottom + right_bottom)/2.0
    deviation_in_pixels = car_centre - lane_centre
    deviation_in_metres = deviation_in_pixels*xm_per_pix
    return left_radius, right_radius, deviation_in_metres


def pipeline(image):
    undistort = undistort_image(image, mtx, dist)
    undistort = gaussian_blur(undistort, kernel=5)
    warp_shape = (undistort.shape[1], undistort.shape[0])
    warped, M_warp, Minv_warp = warp_image(undistort, src, dst, warp_shape)

    yellow_white_binary, image_cmb, image_cmb_color = color_sobel_combined(warped)
    image_cmb_color = gaussian_blur(image_cmb_color, 5)

    left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data = lines_from_histogram(image_cmb_color)

    image_lanes = draw_lane(image, image_cmb_color, left_fit, right_fit, Minv_warp)
    left_radius, right_radius, deviation = \
        radius_of_curvature_and_deviation(image_cmb_color, left_fit, right_fit, left_lane_inds, right_lane_inds)
    str_radius = 'Curvature: Right = ' + str(np.round(right_radius, 2)) + ', Left = ' + str(np.round(left_radius, 2))
    str_deviation = 'Lane deviation: ' + str(deviation) + ' M.'
    cv2.putText(image_lanes, str_radius, (300, 70), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
    cv2.putText(image_lanes, str_deviation, (300, 100), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
    return image_lanes


def pipeline_show_images(image):
    undistort = undistort_image(image, mtx, dist)
    undistort = gaussian_blur(undistort, kernel=5)
    warp_shape = (undistort.shape[1], undistort.shape[0])
    warped, M_warp, Minv_warp = warp_image(undistort, src, dst, warp_shape)

    yellow_white_binary, image_cmb, image_cmb_color = color_sobel_combined(warped)
    image_cmb_color = gaussian_blur(image_cmb_color, 5)

    left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data = lines_from_histogram(image_cmb_color)
    image_lines = draw_lines(image_cmb_color, left_fit, right_fit)

    image_lanes = draw_lane(image, image_cmb_color, left_fit, right_fit, Minv_warp)
    left_radius, right_radius, deviation = \
        radius_of_curvature_and_deviation(image_cmb_color, left_fit, right_fit, left_lane_inds, right_lane_inds)
    str_radius = 'Curvature: Right = ' + str(np.round(right_radius, 2)) + ', Left = ' + str(np.round(left_radius, 2))
    str_deviation = 'Lane deviation: ' + str(deviation) + ' M.'
    cv2.putText(image_lanes, str_radius, (300, 70), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
    cv2.putText(image_lanes, str_deviation, (300, 100), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
    f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=15)
    ax2.imshow(undistort)
    ax2.set_title('Undistorted Image', fontsize=15)
    ax3.imshow(warped)
    ax3.set_title('Warped Image', fontsize=15)
    ax4.imshow(yellow_white_binary, cmap='gray')
    ax4.set_title('Yellow White Binary Image', fontsize=15)
    ax5.imshow(image_cmb, cmap='gray')
    ax5.set_title('Combined', fontsize=15)
    ax6.imshow(image_cmb_color, cmap='gray')
    ax6.set_title('Combined1', fontsize=15)
    ax7.imshow(image_lines, cmap='gray')
    ax7.set_title('Lines on Binary', fontsize=15)
    ax8.imshow(image_lanes)
    ax8.set_title('Lines on original', fontsize=15)
    plt.waitforbuttonpress()


src = np.float32([[545, 460], [735, 460], [1280, 700], [0, 700]])
dst = np.float32([[0, 0], [1280, 0], [1280, 720], [0, 720]])
recent_left_xfitted = deque(maxlen=75)
recent_right_xfitted = deque(maxlen=75)
undistort_mtx_dist = pickle.load(open("../pickle_saved/pickle_un_distortion.p", "rb"))
mtx = undistort_mtx_dist["mtx"]
dist = undistort_mtx_dist["dist"]
# image = mpimg.imread('../test_images/test5.jpg')
# pipeline_show_images(image)

import glob

images = glob.glob("../test_images/*.jpg")
for image in images:
    img = mpimg.imread(image)
    pipeline_show_images(img)

# video_output = '../output_videos/project_video.mp4'
# clip1 = VideoFileClip("../project_video.mp4")
#
# white_clip = clip1.fl_image(pipeline)
# white_clip.write_videofile(video_output, audio=False)
