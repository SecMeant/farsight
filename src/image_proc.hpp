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
  static inline const double box_size = 0.5;
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
            const cv::Mat &imgDepth,
            const bbox &a,
            const farsight::PointArray &flattened);
  void
  calcBiggestComponent(objectType t);
  void
  displayCurrectConfig();
  void
  translate(objectType t);

  void
  saveDepthFrame(int kinectID,
                 const objectType t,
                 const libfreenect2::Frame *frame)
  {
    auto &c = config[kinectID].objects[to_underlying(t)];
    memcpy(c.depthFrame.data, frame->data, depth_width*depth_height*sizeof(float)); 
  }
  void
  setCameraPos(int kinectID, farsight::Point3f pos)
  {
    auto &c = config[kinectID];
    c.camPose = pos;
  }

  double calcMaxDistance()
  {
    auto &c1 = config[0];
    auto &c2 = config[1];
    distance = c1.camPose.z + c2.camPose.z + box_size;
    return distance;
  }

  const bbox&
  getDetectedBox(int kinectID, objectType t)
  {
    return config[kinectID].objects[to_underlying(t)].area;
  }

  const libfreenect2::Frame*
  getDepthFrame(int kinectID, objectType t)
  {
    return &config[kinectID].objects[to_underlying(t)].depthFrame;
  }

  const cv::Mat&
  getBaseDepthImg(int kinectID)
  {
    return config[kinectID].img_base;
  }

  const libfreenect2::Frame*
  getBaseDepthFrame(int kinectID)
  {
    return &config[kinectID].base;
  }

  void
  saveBaseDepthFrame(int kinectID,
                     const libfreenect2::Frame *frame)
  {
    auto &c = config[kinectID];
    memcpy(c.base.data, frame->data, depth_width*depth_height*sizeof(float)); 
  }

  void
  saveBaseDepthImg(int kinectID,
                   const cv::Mat &img)
  {
      auto &c = config[kinectID];
      img.copyTo(c.img_base);
  }

  void
  setNearestPoint(int kinectID, farsight::Point3f &p)
  {
    auto &c = config[kinectID].objects[to_underlying(objectType::MEASURED_OBJ)];
    c.nearest_point = p;
  }

  const farsight::Point3f&
  getNearestPoint(int kinectID)
  {
    return config[kinectID].objects[to_underlying(objectType::MEASURED_OBJ)].nearest_point;
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
  farsight::Point3f cameraOffsets;
  double distance = 0;
};
