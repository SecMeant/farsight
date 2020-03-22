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

const char *wndname = "wnd";
const char *wndname2 = "wnd2";
const char *wndname3 = "wnd3";
const char *wndname4 = "wnd4";

constexpr int avg_config_number = 100;
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
  bbox box_avg;
  depth_t dep_avg;
  dep_avg.depth = 0;
  
  int config_to_go = 1;
  int selectedKinnect = 0;

  detector dec;
  kinect k_dev(selectedKinnect);

  while (continue_flag.test_and_set() and c != 'q')
  {
    k_dev.waitForFrames(10);
    libfreenect2::Frame *rgb   = k_dev.frames[libfreenect2::Frame::Color];
    libfreenect2::Frame *ir    = k_dev.frames[libfreenect2::Frame::Ir];
    libfreenect2::Frame *depth = k_dev.frames[libfreenect2::Frame::Depth];
    if(c == 'c')
    {
        dec.saveOriginalFrameObject(selectedKinnect,depth);
    }
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
              dec.setBaseImg(selectedKinnect,image_depth);
              dec.displayCurrectConfig();
            }
        break;
        case 'c':
            {
              fmt::print("Making {} kinect configuration \n",selectedKinnect+1);
              auto detectedBox = dec.detect(selectedKinnect,
                        depth->data,
                        total_size_depth,
                        image_depth);
              auto minDep = scopeMin<float>(detectedBox,dec.getOriginalFrameObject(selectedKinnect));
              box_avg += detectedBox;
              dep_avg.depth += minDep.depth;

              if( config_to_go < avg_config_number){
                config_to_go++;
                k_dev.releaseFrames();
                continue;
              }

              minDep.depth = dep_avg.depth/avg_config_number;
              detectedBox.w = box_avg.w/avg_config_number;
              detectedBox.y = box_avg.h/avg_config_number;
              dep_avg.depth = 0;
              box_avg.reset();
              config_to_go =1;
              dec.configure(selectedKinnect, image_depth, detectedBox, minDep);
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

    if(dec.isFullyConfigured())
    {
        dec.meassure();
    }

    cv::imshow(wndname, image_rgb);
    cv::imshow(wndname2, image_depth);
    c = cv::waitKey(100);
    k_dev.releaseFrames();
  }
  k_dev.close();
}
