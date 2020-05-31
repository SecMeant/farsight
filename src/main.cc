#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <stdio.h>

#include "3d.h"
#include "camera.h"
#include "filter.h"
#include "image_proc.hpp"
#include "kinect_manager.hpp"
#include "types.h"
#include <chrono>
#include <fmt/ostream.h>
#include <libfreenect2/registration.h>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/photo.hpp>
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
static farsight::postprocessing::Stage1 stage1(depth_width, depth_height);

constexpr int waitTime = 50;
// Defining the dimensions of checkerboard
static int CHECKERBOARD[2]{ 8, 6 };
static cv::Mat cameraMatrix, distCoeffs, R, T;
static std::vector<cv::Vec3d> rvecs, tvecs;
farsight::Point2i interpol[2];
double floor_level = 0.0;

static libfreenect2::Frame depth_frame_cpy =
  libfreenect2::Frame(depth_width, depth_height, sizeof(float));
static bool arucoCalibrated = false;
const char *wndname = "wnd";
const char *wndname2 = "wnd2";
const char *wndname3 = "wnd3";
const char *wndname4 = "wnd4";
const char *wndaruco = "aruco";
const char *arucoConfigPath = "../../../aruco/aruco.conf";

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

double
findFloorLevel(libfreenect2::Registration &reg)
{
  const int x = interpol[0].x;
  const int y_end = interpol[1].y;
  int y_beg = interpol[0].y;

  const int direction = y_beg > y_end ? -1 : 1;
  
  float _1, _2, y, level = std::numeric_limits<double>::max();

  while(y_beg != y_end) {
    reg.getPointXYZ(&depth_frame_cpy, y_beg, x, _1, y, _2);
    if (level > y)
        level = y;
    y_beg += direction;
  }

  return level;
}

void
mouse_event_handler(int event, int x, int y, int flags, void *userdata)
{
  static unsigned int inter_id = 0;
  shared_t *shared = static_cast<shared_t *>(userdata);
  if (event == cv::EVENT_LBUTTONDOWN)
  {
    farsight::Point3f p;
    int pos;
    std::scoped_lock lck(shared->lock);
    shared->reg.getPointXYZ(&depth_frame_cpy, y, x, p.x, p.y, p.z);
    pos = y * depth_width + x;
  fmt::print("{} {}\n", x, y);
    fmt::print(
      "Value: {} {} {} {}\n", depth_frame_cpy.data[pos], p.x, p.y, p.z);
    interpol[inter_id] = {x,y};
    if(inter_id)
    {
        floor_level = findFloorLevel(shared->reg);
        fmt::print("Floor level : {}\n", floor_level);
    }
    fmt::print("point saved under idx = {} \n", inter_id);
    inter_id ^= 0x1;
  }
}

void
calibrateAruco()
{

  // Creating vector to store vectors of 3D points for each checkerboard
  // image
  std::vector<std::vector<cv::Point3f>> objpoints;

  // Creating vector to store vectors of 2D points for each checkerboard
  // image
  std::vector<std::vector<cv::Point2f>> imgpoints;

  // Defining the world coordinates for 3D points
  std::vector<cv::Point3f> objp;
  for (int i{ 0 }; i < CHECKERBOARD[1]; i++)
  {
    for (int j{ 0 }; j < CHECKERBOARD[0]; j++)
      objp.push_back(cv::Point3f(j * 0.0285, i * 0.0285, 0));
  }

  // vector to store the pixel coordinates of detected checker board
  // corners
  std::vector<cv::Point2f> corner_pts;
  bool success;

  // Looping over all the images in the directory
  for (int i{ 0 }; i < images.size(); i++)
  {
    cv::Mat f = cv::imread(images[i]);
    cv::cvtColor(f, f, cv::COLOR_BGR2GRAY);

    // Finding checker board corners
    // If desired number of corners are found in the image then success =
    // true
    success = cv::findChessboardCorners(
      f,
      cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]),
      corner_pts,
      cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

    /*
     * If desired number of corner are detected,
     * we refine the pixel coordinates and display
     * them on the images of checker board
     */
    if (success)
    {
      cv::TermCriteria criteria(
        cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.001);

      // refining pixel coordinates for given 2d points.
      cv::cornerSubPix(
        f, corner_pts, cv::Size(11, 11), cv::Size(-1, -1), criteria);

      // Displaying the detected corner points on the checker board
      cv::drawChessboardCorners(f,
                                cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]),
                                corner_pts,
                                success);

      objpoints.push_back(objp);
      imgpoints.push_back(corner_pts);
    }
  }

  cv::calibrateCamera(objpoints,
                      imgpoints,
                      cv::Size(color_height, color_width),
                      cameraMatrix,
                      distCoeffs,
                      R,
                      T);
  cv::FileStorage storage(arucoConfigPath, cv::FileStorage::WRITE);
  storage << "cameraMatrix" << cameraMatrix;
  storage << "distCoeffs" << distCoeffs;
  storage << "R" << R;
  storage << "T" << T;
  storage.release();
}

void
findAruco(const cv::Mat &f)
{
  // camera parameters are read from somewhere
  cv::Ptr<cv::aruco::Dictionary> dictionary =
    cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_50);
  cv::Mat imageCopy;
  f.copyTo(imageCopy);
  std::vector<int> ids;
  std::vector<std::vector<cv::Point2f>> corners;
  cv::aruco::detectMarkers(f, dictionary, corners, ids);
  // if at least one marker detected
  if (ids.size() > 0)
  {
    cv::aruco::drawDetectedMarkers(imageCopy, corners, ids);
    cv::aruco::estimatePoseSingleMarkers(
      corners, 0.40, cameraMatrix, distCoeffs, rvecs, tvecs);
  if (tvecs.size() && rvecs.size())
  {
    fmt::print("{} {} {}\n", tvecs[0][0],tvecs[0][1],tvecs[0][2] ); 
  }
    // draw axis for each marker
    for (int i = 0; i < ids.size(); i++)
      cv::aruco::drawAxis(
        imageCopy, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.20);
  }
  cv::imshow(wndaruco, imageCopy);
}

void
generateScene(const libfreenect2::Registration &reg,
              const libfreenect2::Frame *f,
              const int cam)
{
  farsight::Point3f p{ 0, 0, 0 };
  farsight::PointArray pointMap;

  if (!tvecs.size() || !rvecs.size())
    return;

  float nan = NAN;

  auto &tvec = tvecs[0];
  glm::vec3 gtvec = { tvec[0], tvec[1], tvec[2] };

  cv::Mat r_mat;
  cv::Rodrigues(rvecs[0], r_mat);
  cv::Mat translation_matrix = r_mat.inv();

  glm::mat3x3 grmat;
  for (int r = 0; r < translation_matrix.rows; r++)
  {
    for (int c = 0; c < translation_matrix.cols; c++)
    {
      auto d = translation_matrix.at<double>(r, c);
      grmat[r][c] = d;
    }
  }

  int pos;
  for (int r = 0; r < f->height; r++)
  {
    for (int c = 0; c < f->width; c++)
    {
      pos = r * f->width + c;
      reg.getPointXYZ(f, r, c, p.x, p.y, p.z);

      if (p.z < 4.5)
      {
        pointMap.push_back({ p.x, p.y, p.z });
      }
      else
      {
        pointMap.push_back({ nan, nan, nan });
      }
    }
  }
  fmt::print("Updating opngl");
  farsight::camera2real(pointMap, gtvec, grmat);
  if (cam == 0)
    farsight::update_points_cam1(pointMap, depth_width);
  else
    farsight::update_points_cam2(pointMap, depth_width);
}

// return array of points with mapped
// the real x y z coordinates in milimiters
farsight::PointArray
createPointMaping(const libfreenect2::Registration &reg,
                  const libfreenect2::Frame *f,
                  const byte *filtered,
                  const bbox &b,
                  int cam,
                  double distance)
{
  farsight::Point3f p{ 0, 0, 0 };
  farsight::PointArray pointMap;
  int pos;

  if (!tvecs.size() || !rvecs.size())
    return {};

  float nan = NAN;

  auto &tvec = tvecs[0];
  glm::vec3 gtvec = { tvec[0], tvec[1], tvec[2] };

  cv::Mat r_mat;
  cv::Rodrigues(rvecs[0], r_mat);
  cv::Mat translation_matrix = r_mat.inv();

  glm::mat3x3 grmat;
  for (int r = 0; r < translation_matrix.rows; r++)
  {
    for (int c = 0; c < translation_matrix.cols; c++)
    {
      auto d = translation_matrix.at<double>(r, c);
      grmat[r][c] = d;
    }
  }
  fmt::print("{} {} {} {}", b.x, b.y, b.w, b.h);
  for (int r = b.y; r < b.y + b.h; r++)
  {
    for (int c = b.x; c < b.x + b.w; c++)
    {
      pos = r * b.w + c;
      if (filtered[pos] == 255)
      {
        pointMap.push_back({ nan, nan, nan });
        continue;
      }

      reg.getPointXYZ(f, r, c, p.x, p.y, p.z);
      if (p.z > distance  || p.y >= floor_level)
      {
        pointMap.push_back({ nan, nan, nan });
        continue;
      }

      pointMap.push_back({ p.x, p.y, p.z });
    }
  }

  farsight::camera2real(pointMap, gtvec, grmat);
  if (cam == 0)
  {
    farsight::update_points_cam1(pointMap, depth_width);
    return farsight::get_translated_points_cam1();
  }

  farsight::update_points_cam2(pointMap, depth_width);
  return farsight::get_translated_points_cam2();
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
  double distance = 0;
  int selectedKinnect = 0;
  constexpr int filterTimes = 100;
  int filterCounter = 0;
  std::thread gl_thread(farsight::init3d);
  gl_thread.detach();

  detector dec;
  kinect k_dev(selectedKinnect);
  libfreenect2::Frame undistorted(
    depth_width, depth_height, sizeof(float)),
    registered(color_width, color_height, sizeof(unsigned int));

  libfreenect2::Registration reg(k_dev.getIRParams(),
                                 k_dev.getColorParams());
  shared_t shared{ std::mutex(), reg };
  cv::namedWindow(wndname2, cv::WINDOW_AUTOSIZE);
  cv::setMouseCallback(wndname2, mouse_event_handler, &shared);

  objectType t;
  byte *depth_backup = nullptr;
  while (continue_flag.test_and_set() and c != 'q')
  {
    k_dev.waitForFrames(10);

    rgb = k_dev.frames[libfreenect2::Frame::Color];
    ir = k_dev.frames[libfreenect2::Frame::Ir];
    depth = k_dev.frames[libfreenect2::Frame::Depth];
    depth_backup = depth->data; 

    if (c == 'r')
    {
      t = objectType::REFERENCE_OBJ;
    }
    else if (c == 'o')
    {
      t = objectType::MEASURED_OBJ;
    }

    if (c == 'b' || c == 'r' || c == '0' || c == 'n')
    {
      stage1.apply(*depth);
      if(filterCounter < filterTimes)
      {
        filterCounter++;
        k_dev.releaseFrames();
        continue;
      }
      filterCounter = 0;
      memcpy(depth_frame_cpy.data,
             stage1.get().data,
             depth_width * depth_height * sizeof(float));
      depth->data = stage1.get().data;
    }

    cv::Mat gray;
    auto image_rgb = cv::Mat(rgb->height, rgb->width, CV_8UC4, rgb->data);
    cv::cvtColor(image_rgb, gray, cv::COLOR_BGR2GRAY);
    if (arucoCalibrated == true)
    {
      findAruco(gray);
      if (c == 's')
      {
        const auto &tvec = tvecs[0];
        farsight::Point3f pos = { tvec[0], tvec[1], tvec[2] };
        dec.setCameraPos(selectedKinnect, pos);
        distance = dec.calcMaxDistance();
        fmt::print("Curent distance is {}\n", distance);
        generateScene(reg, depth, selectedKinnect);
      }
    }
    depthProcess(depth);
    conv32FC1To8CU1(depth->data, depth->height * depth->width);
    auto image_depth =
      cv::Mat(depth->height,depth->width, CV_8UC1, depth->data);

    switch (c)
    {
      case 'b': {
        fmt::print("Setting {} kinect base image \n", selectedKinnect + 1);
        dec.saveBaseDepthImg(selectedKinnect, image_depth);
      }
      break;
      case 'c': {
        cv::glob("../../../aruco/*.jpg", images);
        fmt::print("calibration started");
        calibrateAruco();
        arucoCalibrated = true;
      }
      break;
      case 'l': {
        cv::FileStorage storage(arucoConfigPath, cv::FileStorage::READ);
        storage["cameraMatrix"] >> cameraMatrix;
        storage["distCoeffs"] >> distCoeffs;
        storage["R"] >> R;
        storage["T"] >> T;
        storage.release();
        arucoCalibrated = true;
      }
      break;
      case 'n': {
        auto detectedBox = dec.detect(
          selectedKinnect, depth->data, total_size_depth, image_depth);
        auto nearestPoint = findNearestPoint<float>(
          detectedBox, depth_frame_cpy.data, depth->data);
        dec.setNearestPoint(selectedKinnect, nearestPoint);
        fmt::print("nearest point {}", nearestPoint.z);
      }
      break;
      case 'r':
      case 'o': // find object depth
      {
        dec.saveDepthFrame(selectedKinnect, t, &depth_frame_cpy);
        auto detectedBox = dec.detect(
          selectedKinnect, depth->data, total_size_depth, image_depth);
        const auto &np = dec.getNearestPoint(selectedKinnect == 0 ? 1 : 0);
        double dist = distance - np.z;
        auto realPoints = createPointMaping(reg,
                                            &depth_frame_cpy,
                                            depth->data,
                                            detectedBox,
                                            selectedKinnect,
                                            dist);
        
        dec.setConfig(
          selectedKinnect, t, image_depth, detectedBox, realPoints);
        dec.displayCurrectConfig();
        dec.calcBiggestComponent(t);
      }
      break;
      case '1':
        if (k_dev.open(0))
          selectedKinnect = 0;
        break;
      case '2':
        if (k_dev.open(1))
          selectedKinnect = 1;
        break;
    }

    cv::imshow(wndname2, image_depth);
    c = cv::waitKey(waitTime);
    if (c == 'b' || c == 'n' || c == 'o' || c == 'r')
    {
      std::this_thread::sleep_for(std::chrono::seconds(4));
      stage1.reset();
      depth->data = depth_backup;
    }
    k_dev.releaseFrames();
  }
  k_dev.close();
}
