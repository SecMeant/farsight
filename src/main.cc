#include <memory>
#include <stdio.h>
#include <cmath>

#include <fmt/ostream.h>
#include "image_proc.hpp"
#include "kinect_manager.hpp"
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

int
main(int argc, char **argv)
{
  continue_flag.test_and_set();
  if (signal(SIGINT, sigint_handler) == SIG_ERR)
  {
    fmt::print("Failed to register signal handler.\n");
    exit(-2);
  }
  const size_t depth_width = 512, depth_height = 424;
  const size_t total_size_depth = depth_width*depth_height;
  const size_t rgb_width = 1920, rgb_height = 1080;
  int c = 0;
  
  int selectedKinnect =0;

  std::array<byte, depth_width*depth_height> frame_depth_[2];
  detector dec;
  kinect k_dev(selectedKinnect);

  while (continue_flag.test_and_set() and c != 'q')
  {
    k_dev.waitForFrames(10);
    libfreenect2::Frame *rgb   = k_dev.frames[libfreenect2::Frame::Color];
    libfreenect2::Frame *ir    = k_dev.frames[libfreenect2::Frame::Ir];
    libfreenect2::Frame *depth = k_dev.frames[libfreenect2::Frame::Depth];

    depthProcess(depth);
    
    conv32FC1To8CU1(depth->data , depth->height* depth->width);
 
    auto image_depth =
      cv::Mat(depth->height, depth->width, CV_8UC1, depth->data);
    auto image_rgb = cv::Mat(rgb->height, rgb->width, CV_8UC4, rgb->data);
    
    switch(c)
    {
        case 'b':
            {
              fmt::print("Setting {} kinect base image \n",selectedKinnect+1);
              std::copy(depth->data,
                        depth->data + total_size_depth,
                        frame_depth_[BASE].data());
              auto base_img=
                cv::Mat(depth->height, depth->width, CV_8UC1, frame_depth_[BASE].data());
              dec.setBaseImg(selectedKinnect, base_img);
              dec.displayCurrectConfig();
            }
        break;
        case 'c':
            {
              fmt::print("Making {} kamera configuration \n",selectedKinnect+1);
              std::copy(depth->data,
                        depth->data + total_size_depth,
                        frame_depth_[WITH_OBJECT].data());

              auto detectedBox = dec.detect(frame_depth_[BASE].data(),
                          frame_depth_[WITH_OBJECT].data(),
                          total_size_depth,
                          image_depth);
              dec.configure(selectedKinnect, image_depth, detectedBox);
              dec.displayCurrectConfig();
            }
        break;
        case '1':
            selectedKinnect = 0;
            k_dev.open(selectedKinnect);
        break;
        case '2':
            selectedKinnect = 1;
            k_dev.open(selectedKinnect);
        break;

    }
    cv::imshow(wndname, image_rgb);
    cv::imshow(wndname2, image_depth);
    c = cv::waitKey(100);
    k_dev.releaseFrames();
  }
  k_dev.close();
}
