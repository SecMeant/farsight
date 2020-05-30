#include <cmath>
#include <memory>
#include <stdio.h>
#include <limits>
#include <mutex>

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
#include "3d.h"
#include "camera.h"
#include "types.h"
#include <thread>

extern "C"
{
#include <signal.h>
#include <unistd.h>
}

struct shared_t
{
  std::mutex lock;
  libfreenect2::Registration &reg;
};

constexpr int waitTime=50;
// Defining the dimensions of checkerboard
static int CHECKERBOARD[2]{8,6}; 
static cv::Mat cameraMatrix,distCoeffs,R,T, rvec;
static std::vector<cv::Vec3d> rvecs, tvecs;
static libfreenect2::Frame md_frame = libfreenect2::Frame(depth_width, depth_height, sizeof(float));
static bool arucoCalibrated = false;
const char *wndname  = "wnd";
const char *wndname2 = "wnd2";
const char *wndname3 = "wnd3";
const char *wndname4 = "wnd4";
const char *wndaruco = "aruco";

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
void
mouse_event_handler( int event, int x, int y, int flags, void *userdata )
{
  shared_t *shared = static_cast<shared_t *>(userdata);
  if (event == cv::EVENT_LBUTTONDOWN)
  {
    farsight::Point3f p;
    int pos;
    std::scoped_lock lck(shared->lock);
    shared->reg.getPointXYZ(&md_frame, y, x, p.x, p.y, p.z);
    pos = y*depth_width + x;
    fmt::print("Value: {} {} {} {}\n",md_frame.data[pos], p.x, p.y, p.z);
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
    success = cv::findChessboardCorners(f, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
    
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
  }

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
    cv::aruco::estimatePoseSingleMarkers(corners, 0.40, cameraMatrix, distCoeffs, rvecs, tvecs);
    if(tvecs.size()  && rvecs.size())
    {
        fmt::print("{} {} {} \n", tvecs[0][0], tvecs[0][1],tvecs[0][2]);
    }
    // draw axis for each marker
    for(int i=0; i<ids.size(); i++)
        cv::aruco::drawAxis(imageCopy, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.20);
  }
  cv::imshow(wndaruco, imageCopy);
}

void generateScene(const libfreenect2::Registration &reg,
                     const libfreenect2::Frame *f)
{
 farsight::Point3f p{0,0,0};
 pointArray pointMap;

 if(!tvecs.size() || !rvecs.size())
    return;

 float nan = NAN;

 auto& tvec = tvecs[0];
 glm::vec3 gtvec = {tvec[0], tvec[1], tvec[2]};

 cv::Mat r_mat;
 cv::Rodrigues(rvecs[0], r_mat);
 cv::Mat translation_matrix = r_mat.inv();

 glm::mat3x3 grmat;
 for(int r = 0; r< translation_matrix.rows; r++)
 {
    for( int c = 0; c < translation_matrix.cols; c++)
    {
        auto d = translation_matrix.at<double>(r,c);
        grmat[c][r] = d;
    }
 }

 float *ud = reinterpret_cast<float*>(f->data);
 int pos;
 for(int r=0; r < f->height; r++)
 {
   for(int c=0; c < f->width; c++)
   {
    pos = r*f->width + c;
    reg.getPointXYZ(f, r, c, p.x, p.y, p.z);

    if(p.z < 4.5){
        pointMap.push_back({p.x, p.y, p.z});
    }
    else{
        pointMap.push_back({nan,nan,nan});
    }
   }
 }
 fmt::print("Updating opngl");
 farsight::camera2real(pointMap, gtvec, grmat);
 farsight::update_points_cam1(pointMap, depth_width);
}

// return array of points with mapped 
// the real x y z coordinates in milimiters
pointArray 
createPointMaping(const libfreenect2::Registration &reg,
                  const libfreenect2::Frame *f,
                  const byte *filtered,
                  const bbox &b)
{
 farsight::Point3f p;
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
    map.push_back({p.x*M_TO_MM, p.y*M_TO_MM, p.z*M_TO_MM});
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
  libfreenect2::Frame *rgb, *ir, *depth; 

  int c = 0;
  bbox boxAverage;
  farsight::Point3f nearestPointAvg;
  nearestPointAvg.z= 0;

  int avg_number = 0; // 1?
  int selectedKinnect = 0;
  std::thread gl_thread(farsight::init3d);
  gl_thread.detach();

  detector dec;
  kinect k_dev(selectedKinnect);
  libfreenect2::Frame undistorted(depth_width, depth_height, sizeof(float)), 
                      registered(color_width, color_height, sizeof(unsigned int));

  libfreenect2::Registration reg(k_dev.getIRParams(),
                                 k_dev.getColorParams());
  shared_t shared{std::mutex(), reg};
  cv::namedWindow(wndname2, cv::WINDOW_AUTOSIZE);
  cv::setMouseCallback(wndname2, mouse_event_handler, &shared);
  
  objectType t;
  while (continue_flag.test_and_set() and c != 'q')
  {
    k_dev.waitForFrames(10);

    rgb = k_dev.frames[libfreenect2::Frame::Color];
    ir = k_dev.frames[libfreenect2::Frame::Ir];
    depth = k_dev.frames[libfreenect2::Frame::Depth];
    memcpy(md_frame.data, depth->data, depth_width*depth_height*sizeof(float)); 
    auto image_rgb = cv::Mat(rgb->height, rgb->width, CV_8UC4, rgb->data);
    cv::Mat gray;
    cv::cvtColor(image_rgb, gray,cv::COLOR_BGR2GRAY);
    
    if (c == 'r')
    {
      t = objectType::REFERENCE_OBJ;
      dec.saveDepthFrame(selectedKinnect, t, depth);
    }else if(c == 'o')
    {
      t = objectType::MEASURED_OBJ;
      dec.saveDepthFrame(selectedKinnect, t, depth);
    }

    if(arucoCalibrated == true)
    {
        findAruco(gray);
        if(c == 's')
        {
          generateScene(reg, depth);
        }
    }

    depthProcess(depth);

    conv32FC1To8CU1(depth->data, depth->height * depth->width);

    auto image_depth =
      cv::Mat(depth->height, depth->width, CV_8UC1, depth->data);

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
        cv::glob("../../../aruco/*.jpg", images);
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
          nearestPointAvg.z= 0;
          boxAverage.reset();
          avg_number = 0;
          auto realPoints = createPointMaping(reg, depth, depth->data, detectedBox);
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

    cv::imshow(wndname2, image_depth);
    c = cv::waitKey(waitTime);
    k_dev.releaseFrames();
  }
  k_dev.close();
}
