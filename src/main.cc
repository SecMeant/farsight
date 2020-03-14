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
  cv::SimpleBlobDetector::Params params;
  params.filterByArea = true;
  params.minArea = 5;
  params.maxArea = std::numeric_limits<float>::infinity();

  cv::Ptr<cv::SimpleBlobDetector> det = cv::SimpleBlobDetector::create(params);

  int c = 0;
  int lowerb = 0, higherb = 255;
  int lowerb2 = 20, higherb2 = 240;
  int area = 0;
  bool clr = false;

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
            diff(frame_depth_[WITH_OBJECT].data(), frame_depth_[BASE].data(), depth->height* depth->width);
        break;
    }
    c =0;
    auto image_depth =
      cv::Mat(depth->height, depth->width, CV_8UC1, depth->data);
    auto image_rgb = cv::Mat(rgb->height, rgb->width, CV_8UC4, rgb->data);

    cv::Mat image_depth_th, image_depth_filtered;
    cv::threshold(image_depth, image_depth_th, lowerb, higherb, cv::THRESH_BINARY_INV);
    cv::inRange(image_depth, lowerb2, higherb2, image_depth_filtered);

    cv::Mat labels, stats, centroids;
    cv::connectedComponentsWithStats(image_depth_filtered, labels, stats, centroids);

    clr = !clr;

    struct bbox
    {
      int x, y, w, h, area;
    }best_bbox;

    int barea = 0;
    for (int i = 0; i < stats.rows; ++i)
    {
      int x = stats.at<int>({0,i});
      int y = stats.at<int>({1,i});
      int w = stats.at<int>({2,i});
      int h = stats.at<int>({3,i});
      int area = stats.at<int>({4,i});

      if (area >= depth_width * depth_height / 2)
        continue;

      if (area > best_bbox.area) {
        best_bbox.area = area;
        best_bbox.x = x;
        best_bbox.y = y;
        best_bbox.w = w;
        best_bbox.h = h;
      }
    }

    fmt::print("Area: {}\n", best_bbox.area);
    cv::Scalar color(clr ? 255 : 0, 0, 0);
    cv::Rect rect(best_bbox.x,best_bbox.y,best_bbox.w,best_bbox.h);
    cv::rectangle(image_depth, rect, color, 3);

    cv::imshow(wndname, image_rgb);
    cv::imshow(wndname2, image_depth);
    c = cv::waitKey(100);
    listener.release(frames);
  }
  dev->stop();
  dev->close();
}
