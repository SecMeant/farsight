#pragma once
#include "image_utils.hpp"

class detector
{
 public:
   static inline constexpr int maxKinectCount = 2;
   static inline constexpr size_t depth_width = 512, depth_height = 424;
   static inline constexpr double cubeWidth = 50.0f; 
   struct bbox
   {
       int x =0, y=0, w=0, h=0, area=0;
   };

   struct objectConfig
   {
       cv::Mat img_base;
       cv::Mat img_object;
       bbox area;
       bool imBaseSet = false, imObjectSet= false;
   };

   // public methods
   detector();

   bbox detect(byte* frame_base, byte *frame_object, size_t size, cv::Mat &image_depth);
   void configure(int kinectID, cv::Mat &img, bbox &sizes);
   void setBaseImg(int kinectID, cv::Mat &img);
   void presentResults();
   void displayCurrectConfig();
 private:
   cv::Ptr<cv::SimpleBlobDetector> det;
   cv::Point2i o_center;
   cv::Mat o_cropped;
   std::array<objectConfig, maxKinectCount> sceneConfiguration;
   std::array<objectConfig, maxKinectCount> meassuredObjects;
};


