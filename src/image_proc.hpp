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

   struct depth
   {
    static inline constexpr double maxDepth = 1.0;
    int pixel_x=0, pixel_y=0;
    double depth = maxDepth;
   };

   struct objectConfig
   {
       cv::Mat img_base;
       cv::Mat img_object;
       bbox area;
       depth dep;
       bool imBaseSet = false, imObjectSet= false;

       void findDepth()
       {
        dep.depth = depth::maxDepth; 
        const auto* depth = reinterpret_cast<float*>(img_object.ptr(area.y, area.x));
        for(int i = 0; i < area.w*area.h; i++)
        {
               if(dep.depth > *depth)
               {
                 dep.depth = *depth;
                 dep.pixel_x = area.x + (i/area.w);
                 dep.pixel_y = area.y + (i%area.w);
               }
               depth++;
        }
        dep.depth *= 4500; 
       }
   };

   // public methods
   detector();

   bbox detect(byte* frame_base, byte *frame_object, size_t size, cv::Mat &image_depth);
   void configure(int kinectID, cv::Mat &img, bbox &sizes);
   void setBaseImg(int kinectID, cv::Mat &img);
   void meassure();
   void displayCurrectConfig();
   bool isFullyConfigured();

 private:
   cv::Ptr<cv::SimpleBlobDetector> det;
   cv::Point2i o_center;
   cv::Mat o_cropped;
   std::array<objectConfig, maxKinectCount> sceneConfiguration;
   std::array<objectConfig, maxKinectCount> meassuredObjects;
   cv::Mat configScreen;
   cv::Rect matRoi;

   void presentResults();
};


