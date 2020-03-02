#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/logger.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/registration.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <atomic>
#include <fmt/format.h>
#include <memory>

#include "viewer.h"

std::atomic_flag continue_flag;
const char *window_name = "kinnect_processed_image_rgb";

extern "C"
{
#include <signal.h>
#include <unistd.h>
}

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

void
process_image(Mat &frame)
{}

int
main()
{
  continue_flag.test_and_set();
  namedWindow(window_name, WINDOW_AUTOSIZE);
  if (signal(SIGINT, sigint_handler) == SIG_ERR)
  {
    fmt::print("Failed to register signal handler.\n");
    return -2;
  }

  libfreenect2::Freenect2 freenect2;

  if (freenect2.enumerateDevices() == 0)
  {
    fmt::print("No devices connected\n");
    return -1;
  }

  std::string serial = freenect2.getDefaultDeviceSerialNumber();

  fmt::print("Connecting to the device with serial: {}\n", serial);

  auto pipeline = new libfreenect2::OpenGLPacketPipeline;
  auto dev = freenect2.openDevice(serial, pipeline);

  libfreenect2::SyncMultiFrameListener listener(
    libfreenect2::Frame::Color
    | libfreenect2::Frame::Ir
    | libfreenect2::Frame::Depth
   );
  libfreenect2::FrameMap frames;

  dev->setColorFrameListener(&listener);
  dev->setIrAndDepthFrameListener(&listener);

  libfreenect2::Registration* registration = new libfreenect2::Registration(dev->getIrCameraParams(), dev->getColorCameraParams());
  libfreenect2::Frame undistorted(512, 424, 4), registered(512, 424, 4);

  if (!dev->start())
    return -1;

  fmt::print("Connecting to the device\n"
             "Device serial number	: {}\n"
             "Device firmware	: {}\n",
             dev->getSerialNumber(),
             dev->getFirmwareVersion());

  Viewer viewer;
  viewer.initialize();

  Mat image_rgb, image_depth;
  char sign = '\0';

  while (continue_flag.test_and_set())
  {

    if (!listener.waitForNewFrame(frames, 10 * 1000))
    {
      fmt::print("TIMEDOUT !\n");
      return -1;
    }

    libfreenect2::Frame *rgb = frames[libfreenect2::Frame::Color];
    libfreenect2::Frame *ir = frames[libfreenect2::Frame::Ir];
    libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];

    registration->apply(rgb, depth, &undistorted, &registered);
    fmt::print("Format :{}", rgb->format);
    //if (' ' == sign)
    {
      image_rgb = Mat(rgb->height, rgb->width, CV_8UC4, rgb->data);
      for(int i =0 ; i < depth->height*depth->width; i++)
          *(reinterpret_cast<float*>(&depth->data[i*4]))/=4500;
      image_depth = Mat(depth->height,depth->width, CV_32FC1, depth->data);
      process_image(image_rgb);
      imshow(window_name, image_depth);
    }

    sign = waitKey(1);

    viewer.addFrame("RGB", rgb);
    viewer.addFrame("ir", ir);
    viewer.addFrame("depth", depth);
    viewer.addFrame("registered", &registered);
    if (viewer.render())
      continue_flag.clear();

    listener.release(frames);
  }

  dev->stop();
  dev->close();

  return 0;
}
