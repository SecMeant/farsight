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
  while(c != 'q')
  {
      inputVideo.grab();
      cv::Mat image;
      inputVideo.retrieve(image);
      if( c == 'c')
          cv::imwrite(fmt::format("../../../aruco/frame{}.jpg", frame_nr++),image);
      cv::imshow("out",image);
      c = (char)cv::waitKey(waitTime);
  }
}
