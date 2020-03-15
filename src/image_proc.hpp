#pragma once
#include "image_utils.hpp"

class detector
{
   cv::Ptr<cv::SimpleBlobDetector> det;
   cv::Point2i o_center;
   cv::Mat o_cropped;

 public:
   const size_t depth_width = 512, depth_height = 424;
   detector();

   void detect(byte* frame_base, byte *frame_object, size_t size, cv::Mat &image_depth);
};


