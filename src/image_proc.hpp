#pragma once
#include "image_utils.hpp"
#include "objects.hpp"
#include <cmath>
#include <fmt/format.h>
#include <memory>

class detector
{
public:
  static inline constexpr int maxKinectCount = 2;

  // public methods
  detector();

  bbox
  detect(int kinectID,
         const byte *frame_object,
         size_t size,
         cv::Mat &image_depth);
  void
  setConfig(int kinectID,
            const objectType t,
            const cv::Mat &img,
            const bbox &a,
            const position &p);
  void
  saveDepthFrame(int kinectID,
                 const objectType t,
                 const libfreenect2::Frame *frame);
  void 
  setBaseImg(int kinectID, const cv::Mat &img);

  void
  meassure();
  void
  displayCurrectConfig();

  const byte *
  getDepthFrame(int kinectID, objectType t)
  {
    return config[kinectID].objects[to_underlying(t)].depthFrame.get();
  }

  bool isConfigured()
  {
    return false;
  }

private:
  cv::Ptr<cv::SimpleBlobDetector> det;
  std::array<cameraConfig, maxKinectCount> config;
  cv::Mat configScreen;
  cv::Rect matRoi;

  void
  presentResults();
};
