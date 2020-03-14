#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/logger.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/registration.h>

#include <memory>
#include <stdio.h>

#include <cmath>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include "image_proc.hpp"

extern "C"
{
#include <signal.h>
#include <unistd.h>
}
enum depth_type
{
    BASE,
    WITH_OBJECT,
};
const char *wndname = "wnd";
const char *wndname2 = "wnd2";
const char *wndname3 = "wnd3";
const char *wndname4 = "wnd4";

std::atomic_flag continue_flag;
libfreenect2::SyncMultiFrameListener listener(
  libfreenect2::Frame::Color | libfreenect2::Frame::Ir |
  libfreenect2::Frame::Depth);
libfreenect2::FrameMap frames;
libfreenect2::Freenect2Device *dev;
libfreenect2::Freenect2 freenect2;

using namespace cv;

void
sigint_handler(int signo)
{
  fmt::print("Signal handler\n");

  if (signo == SIGINT)
  {
    fmt::print("Got SIGINT\n");
    continue_flag.clear();
  }
}

void libfreenectInit()
{

  if (freenect2.enumerateDevices() == 0)
  {
    fmt::print("No devices connected\n");
    exit(-1);
  }

  std::string serial = freenect2.getDefaultDeviceSerialNumber();

  fmt::print("Connecting to the device with serial: {}\n", serial);

  auto pipeline = new libfreenect2::OpenGLPacketPipeline;
  dev = freenect2.openDevice(serial, pipeline);

  dev->setColorFrameListener(&listener);
  dev->setIrAndDepthFrameListener(&listener);

  if (!dev->start())
    exit(-1);

  fmt::print("Connecting to the device\n"
             "Device serial number	: {}\n"
             "Device firmware	: {}\n",
             dev->getSerialNumber(),
             dev->getFirmwareVersion());
}

int
main(int argc, char **argv)
{
  continue_flag.test_and_set();
  if (signal(SIGINT, sigint_handler) == SIG_ERR)
  {
    fmt::print("Failed to register signal handler.\n");
    exit(-2);
  }
 
  libfreenectInit();

  const size_t depth_width = 512, depth_height = 424;
  const size_t total_size_depth = depth_width*depth_height;
  const size_t rgb_width = 1920, rgb_height = 1080;
  int c = 0;
  std::array<byte, depth_width*depth_height> frame_depth_[2];

  while (continue_flag.test_and_set())
  {
    if (!listener.waitForNewFrame(frames, 10 * 1000))
    {
      fmt::print("TIMEDOUT !\n");
      exit(-1);
    }

    libfreenect2::Frame *rgb = frames[libfreenect2::Frame::Color];
    libfreenect2::Frame *ir = frames[libfreenect2::Frame::Ir];
    libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];

    depthProcess(depth);
    
    conv32FC1To8CU1(depth->data , depth->height* depth->width);
 
    auto image_depth =
      cv::Mat(depth->height, depth->width, CV_8UC1, depth->data);
    auto image_rgb = cv::Mat(rgb->height, rgb->width, CV_8UC4, rgb->data);
    
    switch(c)
    {
        case 'f':
            {
              fmt::print("Copying first frame... \n");
              std::copy(depth->data,
                        depth->data + total_size_depth,
                        frame_depth_[BASE].data());
              auto preview=
                cv::Mat(depth->height, depth->width, CV_8UC1, frame_depth_[BASE].data());
              cv::imshow("Preview", preview);
            }
        break;
        case 's':
            {
              fmt::print("Copying second dframe... \n");
              std::copy(depth->data,
                        depth->data + total_size_depth,
                        frame_depth_[WITH_OBJECT].data());
              auto preview=
                cv::Mat(depth->height, depth->width, CV_8UC1, frame_depth_[WITH_OBJECT].data());
              cv::imshow("Preview", preview);
            }
        break;
        case 'p':
            fmt::print("lookingfor object");
            detectObject(frame_depth_[BASE].data(),
                         frame_depth_[WITH_OBJECT].data(),
                         total_size_depth,
                         image_depth);
        break;
    }
    c =0;

    cv::imshow(wndname, image_rgb);
    cv::imshow(wndname2, image_depth);
    c = cv::waitKey(100);
    listener.release(frames);
  }
  dev->stop();
  dev->close();
}
