#pragma once
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/logger.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/registration.h>

struct kinect
{
  kinect();
  kinect(int d_idx);
  ~kinect();
  bool
  open(int d_idx);
  bool
  waitForFrames(int sec);
  void
  releaseFrames();
  void
  close();
  
  libfreenect2::Freenect2Device::IrCameraParams
  getIRParams()
  {
    return dev->getIrCameraParams();
  }

  libfreenect2::Freenect2Device::ColorCameraParams
  getColorParams()
  {
    return dev->getColorCameraParams();
  }

  void
  setIRParams(libfreenect2::Freenect2Device::IrCameraParams &params)
  {
    dev->setIrCameraParams(params);
  }

  void 
  setColorParams(libfreenect2::Freenect2Device::ColorCameraParams& params)
  {
    dev->setColorCameraParams(params);
  }

  inline static kinect
  nodev() { return {}; }

  bool isActive = false;
  libfreenect2::SyncMultiFrameListener listener;
  libfreenect2::FrameMap frames;
  libfreenect2::Freenect2Device *dev;
  libfreenect2::Freenect2 freenect2;
};
