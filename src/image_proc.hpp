#pragma once
#include "image_utils.hpp"
#include <cmath>
#include<memory>
#include<fmt/format.h>
class detector
{
 public:
   static inline constexpr int maxKinectCount = 2;
   static inline constexpr size_t depth_width = 512, depth_height = 424;
   static inline constexpr size_t depth_total_size = depth_width*depth_height*sizeof(float);
   static inline constexpr double cubeWidth = 50.0f; 
   struct objectConfig
   {
       std::unique_ptr<byte[]> originalObjectFrame = std::make_unique<byte[]>(depth_total_size);
       cv::Mat img_base;
       cv::Mat img_object;
       bbox area;
       depth_t dep;
       bool imBaseSet = false, imObjectSet= false;
   };

   // public methods
   detector();

   bbox detect(int kinectID, const byte *frame_object, size_t size,cv::Mat &image_depth);
   void configure(int kinectID,const cv::Mat &img,const bbox &sizes, const depth_t &dep);
   void setBaseImg(int kinectID,const cv::Mat &img);
   void saveOriginalFrameObject(int kinectID, const libfreenect2::Frame *frame);
   void meassure();
   void displayCurrectConfig();
   bool isFullyConfigured();
   
   const byte* getOriginalFrameObject(int kinectID){
    return sceneConfiguration[kinectID].originalObjectFrame.get();
   }

 private:
   cv::Ptr<cv::SimpleBlobDetector> det;
   cv::Point2i o_center;
   cv::Mat o_cropped;
   std::array<objectConfig, maxKinectCount> meassuredObjects;
   std::array<objectConfig, maxKinectCount> sceneConfiguration;
   cv::Mat configScreen;
   cv::Rect matRoi;

   void presentResults();
};


