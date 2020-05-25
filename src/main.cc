#include <cmath>
#include <memory>
#include <stdio.h>
#include <limits>

#include "image_proc.hpp"
#include "kinect_manager.hpp"
#include <libfreenect2/registration.h>
#include <fmt/ostream.h>

extern "C"
{
#include <signal.h>
#include <unistd.h>
}

const char *wndname  = "wnd";
const char *wndname2 = "wnd2";
const char *wndname3 = "wnd3";
const char *wndname4 = "wnd4";

constexpr int avg_max_number = 100;
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

// return array of points with mapped 
// the real x y z coordinates in milimiters
pointArray 
createPointMaping(const libfreenect2::Registration &reg,
                  const libfreenect2::Frame *f,
                  const byte *filtered,
                  const bbox &b)
{
 Point3f p;
 pointArray map;
 int pos;
 for(int r=b.y; r < b.y+b.h; r++)
 {
   for(int c=b.x; c < b.x+b.w; c++)
   {
    pos = r*b.w + c;
    if(filtered[pos] == 255)
      continue;

    reg.getPointXYZ(f, r, c, p.x, p.y, p.z);
    map.emplace_back(p.x*M_TO_MM, p.y*M_TO_MM, p.z*M_TO_MM);
   }
 }
 return map;
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
  int c = 0;
  bbox boxAverage;
  Point3f nearestPointAvg;
  nearestPointAvg.z= 0;

  int avg_number = 0; // 1?
  int selectedKinnect = 0;

  detector dec;
  kinect k_dev(selectedKinnect);
  libfreenect2::Frame *undistorted, *registered;
  libfreenect2::Registration reg(k_dev.getIRParams(),
                                 k_dev.getColorParams());
  objectType t;
  while (continue_flag.test_and_set() and c != 'q')
  {
    k_dev.waitForFrames(10);
    libfreenect2::Frame *rgb = k_dev.frames[libfreenect2::Frame::Color];
    libfreenect2::Frame *ir = k_dev.frames[libfreenect2::Frame::Ir];
    libfreenect2::Frame *depth = k_dev.frames[libfreenect2::Frame::Depth];
    if (c == 'r')
    {
      t = objectType::REFERENCE_OBJ;
      reg.apply(rgb, depth, undistorted, registered);
      dec.saveDepthFrame(selectedKinnect, t, undistorted);
    }else if(c == 'o')
    {
      t = objectType::MEASURED_OBJ;
      reg.apply(rgb, depth, undistorted, registered);
      dec.saveDepthFrame(selectedKinnect, t, undistorted);
    }

    depthProcess(depth);

    conv32FC1To8CU1(depth->data, depth->height * depth->width);

    auto image_depth =
      cv::Mat(depth->height, depth->width, CV_8UC1, depth->data);
    auto image_rgb = cv::Mat(rgb->height, rgb->width, CV_8UC4, rgb->data);

    switch (c)
    {
      case 'b': {
        fmt::print("Setting {} kinect base image \n", selectedKinnect + 1);
        dec.setBaseImg(selectedKinnect, image_depth);
        dec.displayCurrectConfig();
      }
      break;
      case 'r':
      case 'o': // find object depth
        {
          auto detectedBox = dec.detect(
            selectedKinnect, depth->data, total_size_depth, image_depth);
          auto *frameDepth = dec.getDepthFrame(selectedKinnect, t);
          auto nearestPoint = findNearestPoint<float>(
            detectedBox, frameDepth, depth->data);
          boxAverage += detectedBox;
          nearestPointAvg.z += nearestPoint.z;

          if (avg_number < avg_max_number)
          {
            avg_number++;
            k_dev.releaseFrames();
            continue;
          }
          nearestPoint.z = nearestPointAvg.z / avg_max_number;
          detectedBox.w = boxAverage.w / avg_max_number;
          detectedBox.y = boxAverage.h / avg_max_number;
          nearestPointAvg.z= 0;
          boxAverage.reset();
          avg_number = 0;
          auto realPoints = createPointMaping(reg, undistorted, depth->data, detectedBox);
          dec.setConfig(selectedKinnect, t, image_depth, detectedBox, nearestPoint, realPoints);
          dec.translate(t);
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
