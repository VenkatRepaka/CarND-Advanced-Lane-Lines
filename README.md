## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, your goal is to write a software pipeline to identify the lane boundaries in a video, but the main output or product we want you to create is a detailed writeup of the project.  Check out the [writeup template](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup.  

Creating a great writeup:
---
A great writeup should include the rubric points as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  If you want to extract more test images from the videos, you can simply use an image writing method like `cv2.imwrite()`, i.e., you can read the video in frame by frame as usual, and for frames you want to save for later you can write to an image file.  

To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `ouput_images`, and include a description in your writeup for the project of what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

The `challenge_video.mp4` video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions.  The `harder_challenge.mp4` video is another optional challenge and is brutal!

If you're feeling ambitious (again, totally optional though), don't stop there!  We encourage you to go out and take video of your own, calibrate your camera and show us how you would implement this project from scratch!

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).


## Camera Calibration
##### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.
The code for this section is available in lane finding folder 
    1. calibration.py
    2. undistortion.py

Cameras introduces radial distortion and tangential distortions. Due to radial distortion straight lines appear curved and tangential distortion is because the camera angle is not parallel to image plane.
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, objp is just a replicated array of coordinates, and objpoints will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. imgpoints will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.
Open cv provides few method which are helpful to compute distortion coefficients.
I have used findChessboardCorners and drawChessboardCorners. These methods help to identify the corners and draw lines along the corners.
If findChessboardCorners finds corners then object points which are 3D points and image points which are 2D points of the corners are captured.
Using these object points and image points to calibrate the camera we get the coefficients to rectify distortion. calibrateCamera is used for this purpose.

Below are few images with corners lines drawn
![calibration11](https://github.com/VenkatRepaka/CarND-Advanced-Lane-Lines/blob/master/chessboard_lines/calibration11.jpg)
![calibration12](https://github.com/VenkatRepaka/CarND-Advanced-Lane-Lines/blob/master/chessboard_lines/calibration12.jpg)
![calibration14](https://github.com/VenkatRepaka/CarND-Advanced-Lane-Lines/blob/master/chessboard_lines/calibration14.jpg)
![calibration16](https://github.com/VenkatRepaka/CarND-Advanced-Lane-Lines/blob/master/chessboard_lines/calibration16.jpg)
![calibration2](https://github.com/VenkatRepaka/CarND-Advanced-Lane-Lines/blob/master/chessboard_lines/calibration2.jpg)
![calibration3](https://github.com/VenkatRepaka/CarND-Advanced-Lane-Lines/blob/master/chessboard_lines/calibration3.jpg)



## Pipeline (single images)
#### 1. Provide an example of a distortion-corrected image.
I have applied the distortion coefficients for the test_images

The source code for test image undistortion is available at ./lanefinding/undistorted_test_images.py

Below are the images showing undistortion for test images
![straight_lines1](https://github.com/VenkatRepaka/CarND-Advanced-Lane-Lines/blob/master/test_images_undistorted/staright_lines1.png)
![straight_lines2](https://github.com/VenkatRepaka/CarND-Advanced-Lane-Lines/blob/master/test_images_undistorted/staright_lines2.png)
![test1](https://github.com/VenkatRepaka/CarND-Advanced-Lane-Lines/blob/master/test_images_undistorted/test1.png)
![test2](https://github.com/VenkatRepaka/CarND-Advanced-Lane-Lines/blob/master/test_images_undistorted/test2.png)
![test3](https://github.com/VenkatRepaka/CarND-Advanced-Lane-Lines/blob/master/test_images_undistorted/test3.png)
![test4](https://github.com/VenkatRepaka/CarND-Advanced-Lane-Lines/blob/master/test_images_undistorted/test4.png)
![test5](https://github.com/VenkatRepaka/CarND-Advanced-Lane-Lines/blob/master/test_images_undistorted/test5.png)
![test6](https://github.com/VenkatRepaka/CarND-Advanced-Lane-Lines/blob/master/test_images_undistorted/test6.png)


#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
I have used below coordinates as source and destination points

```
src_points = np.float32([[490, 482], [810, 482], [1250, 720], [40, 720]])
dest_points = np.float32([[0, 0], [1280, 0], [1250, 720], [40, 720]])
```

I would improve the code later to derive points programatically

The code is available in ./lanefinding/birds_eye_view.py

Below are the images after applying perspective transform
![straight_lines1](https://github.com/VenkatRepaka/CarND-Advanced-Lane-Lines/blob/master/birds_eye_view_with_original/straight_lines1.png)
![straight_lines2](https://github.com/VenkatRepaka/CarND-Advanced-Lane-Lines/blob/master/birds_eye_view_with_original/straight_lines2.png)
![test1](https://github.com/VenkatRepaka/CarND-Advanced-Lane-Lines/blob/master/birds_eye_view_with_original/test1.png)
![test2](https://github.com/VenkatRepaka/CarND-Advanced-Lane-Lines/blob/master/birds_eye_view_with_original/test2.png)
![test3](https://github.com/VenkatRepaka/CarND-Advanced-Lane-Lines/blob/master/birds_eye_view_with_original/test3.png)
![test4](https://github.com/VenkatRepaka/CarND-Advanced-Lane-Lines/blob/master/birds_eye_view_with_original/test4.png)
![test5](https://github.com/VenkatRepaka/CarND-Advanced-Lane-Lines/blob/master/birds_eye_view_with_original/test5.png)
![test6](https://github.com/VenkatRepaka/CarND-Advanced-Lane-Lines/blob/master/birds_eye_view_with_original/test6.png)


#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.

I have used sobel thresholding
I have used S color space in HLS color space. s_thresh_min = 170 and s_thresh_max = 255 as threshold values in S color space
Combined the above images finding the pixels which are non zero
Below are images showing different stages of binary threshold image creation

![straight_lines1](https://github.com/VenkatRepaka/CarND-Advanced-Lane-Lines/blob/master/threshold_images_original/straight_lines1.png)
![straight_lines2](https://github.com/VenkatRepaka/CarND-Advanced-Lane-Lines/blob/master/threshold_images_original/straight_lines2.png)
![test1](https://github.com/VenkatRepaka/CarND-Advanced-Lane-Lines/blob/master/threshold_images_original/test1.png)
![test2](https://github.com/VenkatRepaka/CarND-Advanced-Lane-Lines/blob/master/threshold_images_original/test2.png)
![test3](https://github.com/VenkatRepaka/CarND-Advanced-Lane-Lines/blob/master/threshold_images_original/test3.png)
![test4](https://github.com/VenkatRepaka/CarND-Advanced-Lane-Lines/blob/master/threshold_images_original/test4.png)
![test5](https://github.com/VenkatRepaka/CarND-Advanced-Lane-Lines/blob/master/threshold_images_original/test5.png)
![test6](https://github.com/VenkatRepaka/CarND-Advanced-Lane-Lines/blob/master/threshold_images_original/test6.png)

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
I have used the binary image created above to find the line pixels
1. Identified peaks in histogram
2. Identified non zero pixels in the histogram
3. Used non zero indices to fit in a polynomial equation.
4. Averaged the x intercept positions from 5 previous frames.
Position of the vehicle
I have assumed the position of the car is centre of the image
To find the distance from center of the lanes I have chosen the lowest points in the left and right lanes
    ``` 
    car_position = binary_image.shape[1] / 2
    lane_center_position = (left_line.allx[719] + right_line.allx[719]) / 2
    center_dist = (car_position - lane_center_position) * xm_per_pix
    ```
If the distance from centre is negative then the car is left to centre and if positive the the car is right to lane

Below are positions in test images

![fig1](https://github.com/VenkatRepaka/CarND-Advanced-Lane-Lines/blob/master/test_images_final/Figure_1.png)
![fig2](https://github.com/VenkatRepaka/CarND-Advanced-Lane-Lines/blob/master/test_images_final/Figure_2.png)
![fig3](https://github.com/VenkatRepaka/CarND-Advanced-Lane-Lines/blob/master/test_images_final/Figure_3.png)
![fig4](https://github.com/VenkatRepaka/CarND-Advanced-Lane-Lines/blob/master/test_images_final/Figure_4.png)

Here is the link to [project video](https://github.com/VenkatRepaka/CarND-Advanced-Lane-Lines/blob/master/output_videos/project_video_ouput.mp4)

The code used in the project is at [lane_finding_video_pipeline.py](https://github.com/VenkatRepaka/CarND-Advanced-Lane-Lines/blob/master/lanefinding/lane_finding_video_pipeline.py)
To start with I have created smaller python files with minimal functionality and all the files are [here](https://github.com/VenkatRepaka/CarND-Advanced-Lane-Lines/tree/master/lanefinding)
Almost all of the above code pieces are extract from this course and udacity forums. I need to explore few more techniques.

#### Challenges
1. Only S color space is not sufficient to create a threshold binary. Have to evaluate multiple color spaces.
2. I have chosen fixed position to change perspective transform. This led to deviation of the found lines from the actual lines. We should be able to pin point the transformation points programatically
3. I did not perform any smoothing techniques. Smoothing techniques will help at varying contrast and lighting conditions in the video
4. Image resizing should help with speed of video transformation. I have used the same resolution.

