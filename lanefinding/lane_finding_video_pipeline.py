import cv2
import numpy as np
from collections import deque
import pickle
from moviepy.editor import *


src_points = np.float32([[490, 482], [810, 482], [1250, 720], [40, 720]])
dest_points = np.float32([[0, 0], [1280, 0], [1250, 720], [40, 720]])
undistort_mtx_dist = pickle.load(open("../pickle_saved/pickle_un_distortion.p", "rb"))
mtx = undistort_mtx_dist["mtx"]
dist = undistort_mtx_dist["dist"]


class Line:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        # self.recent_xfitted = []
        self.recent_xfitted = deque(maxlen=100)
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        # width of window while detecting pixels of line
        self.margin = 50
        # Minimum number of pixels found to recenter window
        self.minpix = 10
        # Number of windows to be searched for
        self.nwindows = 9

    def radius_of_curvature(self):
        ym_per_pix = 30. / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meteres per pixel in x dimension
        fit_curvature = np.polyfit(self.ally * ym_per_pix, self.allx * xm_per_pix, 2)
        curvature_radius = ((1 + (2 * fit_curvature[0] * np.max(self.allx) + fit_curvature[1]) ** 2) ** 1.5) \
                   / np.absolute(2 * fit_curvature[0])
        return curvature_radius

    def histogram_edges(self, binary_warped):
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        window_height = np.int(binary_warped.shape[0] / self.nwindows)
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        leftx_current = leftx_base
        rightx_current = rightx_base
        left_lane_inds = []
        right_lane_inds = []
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        if np.sum(left_lane_inds) != 0:
            self.detected = True
        else:
            self.detected = False
        return leftx, lefty, rightx, righty

    # def detect_edges_prev_values(self):


left_line = Line()
right_line = Line()


def birds_eye_view(img):
    rows, cols, depth = img.shape

    M = cv2.getPerspectiveTransform(src_points, dest_points)
    warped = cv2.warpPerspective(img, M, (cols, rows), flags=cv2.INTER_LINEAR)

    return warped


def find_lines(binary_warped):
    leftx, lefty, rightx, righty = left_line.histogram_edges(binary_warped)

    # Fit a second order polynomial to each
    # Find new coefficients with average left and right fits
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    left_line.current_fit = left_fit
    right_line.current_fit = right_fit

    # Find left and right x intercepts with the coefficients
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Add the x intercepts to the last intercepts
    left_line.recent_xfitted.append(left_fitx)
    right_line.recent_xfitted.append(right_fitx)

    # Average the left and right fits
    left_fit_avg = np.mean(left_line.recent_xfitted, 0)
    right_fit_avg = np.mean(right_line.recent_xfitted, 0)

    # Randomly choose 100 x points

    # Find new coefficients with the last n average iterations
    left_line.best_fit = np.polyfit(ploty, left_fit_avg, 2)
    right_line.best_fit = np.polyfit(ploty, right_fit_avg, 2)


def create_final_image(image, binary_warped):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_line.best_fit[0] * ploty ** 2 + left_line.best_fit[1] * ploty + left_line.best_fit[2]
    right_fitx = right_line.best_fit[0] * ploty ** 2 + right_line.best_fit[1] * ploty + right_line.best_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    src = np.float32([[490, 482], [810, 482], [1250, 720], [40, 720]])
    dst = np.float32([[0, 0], [1280, 0], [1250, 720], [40, 720]])
    Minv = cv2.getPerspectiveTransform(dst, src)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (binary_warped.shape[1], binary_warped.shape[0]))
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    return result


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


def generate_binary_image(image):
    ksize = 7
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(30, 180))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(30, 180))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(80, 150))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    # Threshold color channel
    s_thresh_min = 170
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    combined_binary = np.zeros_like(s_binary)
    combined_binary[(s_binary == 1) | (combined == 1)] = 1
    return combined_binary


def pipeline(image):
    rows, cols, depth = image.shape
    undistorted_image = cv2.undistort(image, mtx, dist, None, mtx)
    M = cv2.getPerspectiveTransform(src_points, dest_points)
    warped_image = cv2.warpPerspective(undistorted_image, M, (cols, rows), flags=cv2.INTER_LINEAR)
    binary_image = generate_binary_image(warped_image)
    find_lines(binary_image)
    result = create_final_image(image, binary_image)
    return result


# test_image = mpimg.imread("../test_images/test5.jpg")
# pipeline(test_image)

video_output = '../output_videos/challenge_video_ouput.mp4'
clip1 = VideoFileClip("../challenge_video.mp4")

white_clip = clip1.fl_image(pipeline)
white_clip.write_videofile(video_output, audio=False)
