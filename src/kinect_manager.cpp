#include "kinect_manager.hpp"
#include <fmt/format.h>

kinect::kinect()
  : listener(libfreenect2::Frame::Color | libfreenect2::Frame::Ir |
             libfreenect2::Frame::Depth)
{
  this->open(0);
}

kinect::kinect(int d_idx)
  : listener(libfreenect2::Frame::Color | libfreenect2::Frame::Ir |
             libfreenect2::Frame::Depth)
{
  this->open(d_idx);
}

bool
kinect::open(int d_idx)
{
  int devNumber = freenect2.enumerateDevices();
  if (devNumber == 0)
  {
    fmt::print("No devices connected\n");
    exit(-1);
  }
  if (devNumber-1 < d_idx)
      return false;

  fmt::print("Number of found devices: {}", devNumber);
  
  if (isActive)
  {
    this->releaseFrames();
    this->close();
  }
  isActive = true;

  std::string serial = freenect2.getDeviceSerialNumber(d_idx);

  fmt::print("Connecting to the device with serial: {}\n", serial);

  auto pipeline = new libfreenect2::OpenGLPacketPipeline;
  dev = freenect2.openDevice(serial, pipeline);

  dev->setColorFrameListener(&listener);
  dev->setIrAndDepthFrameListener(&listener);

  if (!dev->start())
    return false;

  fmt::print("Connecting to the device\n"
             "Device serial number	: {}\n"
             "Device firmware	: {}\n",
             dev->getSerialNumber(),
             dev->getFirmwareVersion());
  return true;
}

bool
kinect::waitForFrames(int sec)
{
  return listener.waitForNewFrame(frames, sec * 1000);
}

void
kinect::releaseFrames()
{
  listener.release(frames);
}

void
kinect::close()
{
  isActive = false;
  dev->stop();
  dev->close();
}

kinect::~kinect()
{
  if (isActive == true)
  {
    this->close();
  }
}
