#include <opencv2/aruco.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <fmt/format.h>
#include "kinect_manager.hpp"
int CHECKERBOARD[2]{8,6}; 

constexpr int waitTime=50;
int frame_nr =0;
// Defining the dimensions of checkerboard
int main()
{
  char c = ' ';
  std::vector<cv::Point2f> corner_pts;
  bool success;
  kinect k_dev(0);
  bool saveNext = false;
    
  while(c != 'q')
  {
    k_dev.waitForFrames(10);
    libfreenect2::Frame *rgb = k_dev.frames[libfreenect2::Frame::Color];
    libfreenect2::Frame *ir = k_dev.frames[libfreenect2::Frame::Ir];
    libfreenect2::Frame *depth = k_dev.frames[libfreenect2::Frame::Depth];
    auto image = cv::Mat(rgb->height, rgb->width, CV_8UC4, rgb->data);
    cv::cvtColor( image,image,cv::COLOR_BGR2GRAY);
    if ( c == 'c')
        saveNext = true;
    if(saveNext)
    {
          cv::Mat f = image.clone();
          // Finding checker board corners
          // If desired number of corners are found in the image then success = true  
          success = cv::findChessboardCorners(f, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
          if(success)
          {
            cv::TermCriteria criteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.001);
            
            // refining pixel coordinates for given 2d points.
            cv::cornerSubPix(f,corner_pts,cv::Size(11,11), cv::Size(-1,-1),criteria);
            
            // Displaying the detected corner points on the checker board
            cv::drawChessboardCorners(f, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, success);
            cv::imwrite(fmt::format("../../../aruco/frame{}.jpg", frame_nr++),image);
           saveNext = false; 
           image = f;
        }
    }
    k_dev.releaseFrames();
    cv::imshow("out",image);
    c = (char)cv::waitKey(waitTime);
  }
}
