#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/logger.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/registration.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <atomic>
#include <fmt/format.h>
#include <memory>

std::atomic_flag continue_flag;
const char *wnd_rgb = "wndrgb";
const char *wnd_depth = "wnddepth";
const char *wnd_diff= "wnddiff";

extern "C"
{
#include <signal.h>
#include <unistd.h>
}
constexpr int DIFF_FRAMES_COUNT = 2;
enum frames_idx
{
    frame_base_image,
    frame_with_cube,
};

std::array<cv::Mat, DIFF_FRAMES_COUNT> diff_frames;

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
rgb_process(libfreenect2::Frame *frame)
{}

void
depth_process(libfreenect2::Frame *frame)
{
  auto total_size = frame->height * frame->width;
  auto fp = reinterpret_cast<float *>(frame->data);

  for (int i = 0; i < total_size; i++)
  {
    fp[i] /= 4500.0f;
  }
}

void save_frame(const char* fname, libfreenect2::Frame *frame)
{
  auto total_size = frame->height * frame->width;
  auto fp = reinterpret_cast<float *>(frame->data);
  auto fd = fopen(fname, "wb");
  for (int i = 0; i < total_size; i++)
  {
    fwrite(reinterpret_cast<char*>(&fp[i]),4,1,fd);
  }
  fclose(fd);
}

void process_frame(char sign, libfreenect2::Frame *depth)
{
    switch(sign)
    {
        case 'f':
            diff_frames[frames_idx::frame_base_image]= Mat(depth->height, depth->width, CV_32FC1, depth->data);
            fmt::print("taking first snap");
            break;
        case 's':
            diff_frames[frames_idx::frame_with_cube]= Mat(depth->height, depth->width, CV_32FC1, depth->data);
            fmt::print("taking second snap");
            break;
        case 'c':
            {
                fmt::print("Calculating frames");
                cv::Mat thresh;
                auto diff_res = diff_frames[frames_idx::frame_base_image] - diff_frames[frames_idx::frame_with_cube];
                
                cv::threshold(diff_res,thresh, 0.3f, 1.0f, cv::THRESH_BINARY);
                imshow(wnd_diff,thresh);
            }
            break;
    }
}

int
main()
{
  continue_flag.test_and_set();

  namedWindow(wnd_rgb, WINDOW_AUTOSIZE);
  namedWindow(wnd_depth, WINDOW_AUTOSIZE);

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
    libfreenect2::Frame::Color | libfreenect2::Frame::Ir |
    libfreenect2::Frame::Depth);
  libfreenect2::FrameMap frames;

  dev->setColorFrameListener(&listener);
  dev->setIrAndDepthFrameListener(&listener);

  if (!dev->start())
    return -1;

  fmt::print("Connecting to the device\n"
             "Device serial number	: {}\n"
             "Device firmware	: {}\n",
             dev->getSerialNumber(),
             dev->getFirmwareVersion());

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

    rgb_process(rgb);
    image_rgb = Mat(rgb->height, rgb->width, CV_8UC4, rgb->data);

    depth_process(depth);
    image_depth = Mat(depth->height, depth->width, CV_32FC1, depth->data);
    process_frame(sign, depth);
    
    imshow(wnd_rgb, image_rgb);
    imshow(wnd_depth, image_depth);

    sign = waitKey(1);

    listener.release(frames);
  }

  dev->stop();
  dev->close();

  return 0;
}
