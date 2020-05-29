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

constexpr int waitTime=50;
int frame_nr =0;
// Defining the dimensions of checkerboard
int main()
{
  cv::VideoCapture inputVideo;
  inputVideo.open(0); 
  char c;
  bool success = false;
  while(c != 'q')
  {
      inputVideo.grab();
      cv::Mat image;
      inputVideo.retrieve(image);
      if( c == 'c')
      {
          while(not success)
          {
              cv::Mat f = cv::imread(images[i]);
              cv::cvtColor(f,f,cv::COLOR_BGR2GRAY);

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
            }
          }
      }
      cv::imshow("out",image);
      c = (char)cv::waitKey(waitTime);
}
