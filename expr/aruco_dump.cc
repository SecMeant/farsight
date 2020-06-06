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

void
depthProcess(libfreenect2::Frame *frame)
{
  auto total_size = frame->height * frame->width;
  auto fp = reinterpret_cast<float *>(frame->data);

  for (int i = 0; i < total_size; i++)
  {
    fp[i] /= 65535.0f;
  }
}
void
conv32FC1To8CU1(unsigned char *data, size_t size)
{
  auto fp = reinterpret_cast<float *>(data);

  for (auto i = 0; i < size; ++i, ++fp, ++data)
    *data = static_cast<unsigned char>(*fp * 255.0f);
}


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
    
    depthProcess(ir);
    conv32FC1To8CU1(ir->data, ir->height*ir->width);
    auto image =
      cv::Mat(ir->height, ir->width, CV_8UC1, ir->data);

    if ( c == 'c')
        saveNext = true;
    fmt::print("asdkjasjlkdjsalkjdsa");
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
            cv::imwrite(fmt::format("../../../aruco_ir/frame{}.jpg", frame_nr++),image);
           saveNext = false; 
           image = f;
        }
    }
    k_dev.releaseFrames();
    cv::imshow("out",image);
    c = (char)cv::waitKey(waitTime);
  }
}
