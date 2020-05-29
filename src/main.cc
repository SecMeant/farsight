#include <cmath>
#include <memory>
#include <stdio.h>
#include <limits>

#include "image_proc.hpp"
#include "kinect_manager.hpp"
#include <libfreenect2/registration.h>
#include <fmt/ostream.h>
#include <opencv2/aruco.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

extern "C"
{
#include <signal.h>
#include <unistd.h>
}
constexpr int waitTime=50;
// Defining the dimensions of checkerboard
int CHECKERBOARD[2]{6,9}; 
cv::Mat cameraMatrix,distCoeffs,R,T, rvec;
bool arucoCalibrated = false;
const char *wndname  = "wnd";
const char *wndname2 = "wnd2";
const char *wndname3 = "wnd3";
const char *wndname4 = "wnd4";

constexpr int avg_max_number = 10;
std::atomic_flag continue_flag;
std::vector<cv::String> images;
std::vector<cv::Mat> arucoDict;
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
void calibrateAruco()
{

  // Creating vector to store vectors of 3D points for each checkerboard image
  std::vector<std::vector<cv::Point3f> > objpoints;

  // Creating vector to store vectors of 2D points for each checkerboard image
  std::vector<std::vector<cv::Point2f> > imgpoints;
  
  // Defining the world coordinates for 3D points
  std::vector<cv::Point3f> objp;
  for(int i{0}; i<CHECKERBOARD[1]; i++)
  {
    for(int j{0}; j<CHECKERBOARD[0]; j++)
      objp.push_back(cv::Point3f(j*0.0285,i*0.0285,0));
  }

  // vector to store the pixel coordinates of detected checker board corners 
  std::vector<cv::Point2f> corner_pts;
  bool success;
 
  // Looping over all the images in the directory
  for(int i{0}; i<images.size(); i++)
  {
      cv::Mat f = cv::imread(images[i]);
    cv::cvtColor(f,f,cv::COLOR_BGR2GRAY);

    // Finding checker board corners
    // If desired number of corners are found in the image then success = true  
    success = cv::findChessboardCorners(f, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
    
    /* 
     * If desired number of corner are detected,
     * we refine the pixel coordinates and display 
     * them on the images of checker board
    */
    if(success)
    {
      cv::TermCriteria criteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.001);
      
      // refining pixel coordinates for given 2d points.
      cv::cornerSubPix(f,corner_pts,cv::Size(11,11), cv::Size(-1,-1),criteria);
      
      // Displaying the detected corner points on the checker board
      cv::drawChessboardCorners(f, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, success);
      
      objpoints.push_back(objp);
      imgpoints.push_back(corner_pts);
    }

    cv::imshow("Image", f);
    cv::waitKey(10);
  }

  cv::destroyAllWindows();

  cv::calibrateCamera(objpoints, imgpoints, cv::Size(color_height, color_width), cameraMatrix, distCoeffs, R, T);

  std::cout << "cameraMatrix : " << cameraMatrix << std::endl;
  std::cout << "distCoeffs : " << distCoeffs << std::endl;
  std::cout << "Rotation vector : " << R << std::endl;
  std::cout << "Translation vector : " << T << std::endl;
}

void findAruco(const cv::Mat &f)
{
  // camera parameters are read from somewhere
  cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_50);
      cv::Mat imageCopy;
      f.copyTo(imageCopy);
      std::vector<int> ids;
      std::vector<std::vector<cv::Point2f>> corners;
      cv::aruco::detectMarkers(f, dictionary, corners, ids);
      // if at least one marker detected
      if (ids.size() > 0) {
          cv::aruco::drawDetectedMarkers(imageCopy, corners, ids);
          std::vector<cv::Vec3d> rvecs, tvecs;
          cv::aruco::estimatePoseSingleMarkers(corners, 40, cameraMatrix, distCoeffs, rvecs, tvecs);
          if(tvecs.size()  && rvecs.size())
          {
              cv::Rodrigues(rvecs[0], rvec);
              auto v = rvec*cv::Mat(tvecs[0]);
              std::cout<<v<<"\n";
          }
          // draw axis for each marker
          for(int i=0; i<ids.size(); i++)
              cv::aruco::drawAxis(imageCopy, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1);
      }
      cv::imshow("out", imageCopy);
}

void saveFrameToFile(const libfreenect2::Registration &reg,
                     const libfreenect2::Frame *f)
{
 Point3f p;
 auto *file = fopen("frame_decded", "w");
 if(file == nullptr)
     return;

 for(int r=0; r < f->height; r++)
 {
   for(int c=0; c < f->width; c++)
   {
    reg.getPointXYZ(f, r, c, p.x, p.y, p.z);
    fprintf(file, "%f %f %f\n", p.x, p.y, p.z);
   }
 }
 fclose(file);
}
int frame_nr=0;
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
  libfreenect2::Frame undistorted(depth_width, depth_height, sizeof(float)), 
                      registered(color_width, color_height, sizeof(unsigned int));
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
      reg.apply(rgb, depth, &undistorted, &registered);
      dec.saveDepthFrame(selectedKinnect, t, &undistorted);
    }else if(c == 'o')
    {
      t = objectType::MEASURED_OBJ;
      reg.apply(rgb, depth, &undistorted, &registered);
      dec.saveDepthFrame(selectedKinnect, t, &undistorted);
    }

    depthProcess(depth);

    conv32FC1To8CU1(depth->data, depth->height * depth->width);

    auto image_depth =
      cv::Mat(depth->height, depth->width, CV_8UC1, depth->data);
    auto image_rgb = cv::Mat(rgb->height, rgb->width, CV_8UC4, rgb->data);
    cv::Mat gray;
    cv::cvtColor( image_rgb, gray,cv::COLOR_BGR2GRAY);

    switch (c)
    {
      case 'b': {
        fmt::print("Setting {} kinect base image \n", selectedKinnect + 1);
        dec.setBaseImg(selectedKinnect, image_depth);
        dec.displayCurrectConfig();
      }
      break;
      case ' ':
      {
        cv::glob("../../aruco/*.jpg", images);
      }
      break;
      case 'c':
      {
       fmt::print("calibration started");
       calibrateAruco(); 
       arucoCalibrated = true;
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
          saveFrameToFile(reg, &undistorted);
          nearestPointAvg.z= 0;
          boxAverage.reset();
          avg_number = 0;
          auto realPoints = createPointMaping(reg, &undistorted, depth->data, detectedBox);
          dec.setConfig(selectedKinnect, t, image_depth, detectedBox, nearestPoint, realPoints);
          dec.translate(t);
          dec.displayCurrectConfig();
          dec.calcBiggestComponent();
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
    if(arucoCalibrated == true){
        findAruco(gray);
    }

    cv::imshow(wndname,gray);
    cv::imshow(wndname2, image_depth);
    c = cv::waitKey(waitTime);
    k_dev.releaseFrames();
  }
  k_dev.close();
}
