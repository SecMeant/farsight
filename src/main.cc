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
#include "disjoint_set.h"
# define M_PI           3.14159265358979323846
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
static std::vector<int> ids;
static DisjointSet classifier;

constexpr int waitTime = 50;

const int floor_level_max = 1000;
static int floor_level_raw = 0;
static int disjointTreshold = 0;
static int disjointSetValidSize= 0;
// Defining the dimensions of checkerboard
static int CHECKERBOARD[2]{ 8, 6 };
static cv::Mat cameraMatrix, distCoeffs;
static cv::Mat cameraMatrixIR, distCoeffsIR;
static std::vector<cv::Vec3d> rvecs, tvecs;
farsight::Point2i interpol[2];
double floor_level = 1000.0;

static libfreenect2::Frame depth_frame_cpy =
  libfreenect2::Frame(depth_width, depth_height, sizeof(float));
static bool arucoCalibrated = false;
const char *wndname = "wnd";
const char *wndname2 = "wnd2";
const char *wndname3 = "wnd3";
const char *wndname4 = "wnd4";
const char *wndaruco = "aruco";
const char *arucoConfigPath = "../../../aruco/aruco.conf";
static const std::vector<char> base_scenario = { '1', 'b', '2',
                                                 'b', 'e' };
static const std::vector<char> meassure_scenario = { '1', 'n', '2', 'n',
                                                     '1', 'r', '2', 'r',
                                                      'e' };
glm::vec3 cam1_tvec = {0,0,0}, cam2_tvec = {0,0,0}, cam1_rvec = {0,0,0}, cam2_rvec = {0,0,0}; 
std::atomic_flag continue_flag;
std::vector<cv::String> images;
std::vector<cv::String> images_ir;
std::vector<cv::Mat> arucoDict;
std::vector<std::vector<cv::Point2f>> corners;
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

  while (y_beg != y_end)
  {
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
  }
}

void
calibrateArucoColor()
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
  cv::Mat R, T;
  cv::calibrateCamera(objpoints,
                      imgpoints,
                      cv::Size(color_height, color_width),
                      cameraMatrix,
                      distCoeffs,
                      R,
                      T);
}

void
calibrateArucoIr()
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
  for (int i{ 0 }; i < images_ir.size(); i+=5)
  {
    cv::Mat f = cv::imread(images_ir[i]);
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

  cv::Mat R, T;
  cv::calibrateCamera(objpoints,
                      imgpoints,
                      cv::Size(depth_height, depth_width),
                      cameraMatrixIR,
                      distCoeffsIR,
                      R,
                      T);
}

void
findAruco(const cv::Mat &f_)
{
  cv::Mat f, f2;
  cv::cvtColor(f_, f2, cv::COLOR_BGRA2BGR);
  cv::cvtColor(f2, f, cv::COLOR_BGR2GRAY);

  // camera parameters are read from somewhere
  cv::Ptr<cv::aruco::Dictionary> dictionary =
    cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_50);
  cv::Ptr<cv::aruco::DetectorParameters> params = cv::aruco::DetectorParameters::create();
  params->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
  cv::aruco::detectMarkers(f, dictionary, corners, ids, params);
  // if at least one marker detected
  if (ids.size() > 0)
  {
    cv::aruco::drawDetectedMarkers(f2, corners, ids);
    cv::aruco::estimatePoseSingleMarkers(
      corners, 0.40, cameraMatrix, distCoeffs, rvecs, tvecs);
   // if (tvecs.size() && rvecs.size())
   // {
   //   fmt::print("{} {} {}\n", tvecs[0][0], tvecs[0][1], tvecs[0][2]);
   // }
    // draw axis for each marker
    for (int i = 0; i < ids.size(); i++)
      cv::aruco::drawAxis(
        f2, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.20);
  }
  cv::imshow(wndaruco, f2);
}

glm::vec3
findCameraOffsetDiff(const libfreenect2::Registration &reg,
                     const libfreenect2::Frame *f,
                     glm::vec3 gtvec)
{
  const auto &first_face_corners = corners[0];
  const auto &corner = first_face_corners[0];

  int pos;
  float rgb_x, rgb_y;
  farsight::Point3f p;
  auto *data = reinterpret_cast<float *>(f->data);
  for (int r = 0; r < f->height; r++)
  {
    for (int c = 0; c < f->width; c++)
    {
      pos = r * f->width + c;
      reg.apply(r, c, data[pos], rgb_x, rgb_y);
      if(std::isinf(rgb_x) || std::isinf(rgb_y))
          continue;

      if (corner.x < rgb_x && corner.y < rgb_y)
      {
        reg.getPointXYZ(f, r, c, p.x, p.y, p.z);
        auto fix_x = gtvec.x - (gtvec.x - p.x) - 0.25;
        auto fix_y = gtvec.y - (gtvec.y - p.y);
        auto fix_z = gtvec.z;
        fmt::print("FIX OFFSET {} {} {} {} {}\n", corner.x, corner.y, p.x-0.25, p.y, p.z);
        return { fix_x, fix_y, fix_z };
      }
    }
  }
  return {};
}

void
generateScene(const libfreenect2::Registration &reg,
              const libfreenect2::Frame *f,
              const farsight::Point3f &tvec,
              const farsight::Point3f &rvec,
              const int cam)
{
  farsight::Point3f p{ 0, 0, 0 };
  farsight::PointArray pointMap;

  if (!tvecs.size() || !rvecs.size())
    return;

  glm::vec3 gtvec = { tvec.x, tvec.y, tvec.z };
  cv::Vec3d rvec3d  = { rvec.x, rvec.y, rvec.z };
  cv::Mat r_mat;
  cv::Rodrigues(rvec3d, r_mat);
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
        pointMap.push_back({NAN, NAN, NAN});
      }
    }
  }
  fmt::print("Updating opengl\n");
  fmt::print("tvec {} {} {} \n", gtvec.x, gtvec.y, gtvec.z);
  farsight::camera2real(pointMap, gtvec, grmat, ids[0]);
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
                  const farsight::Point3f &tvec,
                  const farsight::Point3f &rvec,
                  const int id,
                  int cam,
                  double distance)
{
  farsight::Point3f p{ 0, 0, 0 };
  classifier.reset();

  glm::vec3 gtvec = { tvec.x, tvec.y, tvec.z };
  cv::Vec3d rvec3d  = { rvec.x, rvec.y, rvec.z };

  cv::Mat r_mat;
  cv::Rodrigues(rvec3d, r_mat);
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
  farsight::PointArray pointMap;
  for (size_t r = b.y; r < b.y + b.h; r++)
  {
    for (size_t c = b.x; c < b.x + b.w; c++)
    {
      pos = r * b.w + c;
      reg.getPointXYZ(f, r, c, p.x, p.y, p.z);
      pointMap.push_back({ p.x, p.y, p.z });
    }
  }

  farsight::camera2real(pointMap, gtvec, grmat, id);
  if (cam == 0)
  {
    farsight::set_tvec_cam1(cam1_tvec);
    farsight::set_rvec_cam1(cam1_rvec);
    farsight::update_points_cam1(pointMap, depth_width);
    pointMap = farsight::get_translated_points_cam1();
  }
  else
  {
    farsight::set_tvec_cam2(cam2_tvec);
    farsight::set_rvec_cam2(cam2_rvec);
    farsight::update_points_cam2(pointMap, depth_width);
    pointMap = farsight::get_translated_points_cam2();
  }
  for(auto &p : pointMap)
  {
    if(p.z > distance)
        classifier.addPoint({NAN,NAN,NAN});
    else
        classifier.addPoint(p); 
  }

  auto cat_sizes = classifier.countCategories();
  pointMap = classifier.getPointsByDelimiter(cat_sizes);

  if (cam == 0)
  {
    farsight::set_tvec_cam1({0,0,0});
    farsight::set_rvec_cam1({0,0,0});
    farsight::update_points_cam1(pointMap, depth_width);
  }
  else
  {
    farsight::set_tvec_cam2({0,0,0});
    farsight::set_rvec_cam2({0,0,0});
    farsight::update_points_cam2(pointMap, depth_width);
  }
  
  return pointMap;
}

static void
on_trackbar(int, void *)
{
  floor_level = double(floor_level_raw) / floor_level_max;
  farsight::set_floor_level(floor_level);
  fmt::print("CURRENT FLOOR LEVEL {}\n", floor_level);
}

static void
on_disjoint_treshold(int, void *)
{
    classifier.updateTreshold(disjointTreshold/100.0);
}

static void
on_disjoint_valid_size(int, void *)
{
    classifier.updateValidSize(disjointSetValidSize);
}

void calibrateCamera(kinect &dev)
{
  auto ir_params =  dev.getIRParams();
  auto color_params =  dev.getColorParams();

  color_params.fx = cameraMatrix.at<double>(0,0);
  color_params.fy = cameraMatrix.at<double>(1,1);
  color_params.cx = cameraMatrix.at<double>(0,2);
  color_params.cy = cameraMatrix.at<double>(1,2);

  ir_params.fx = cameraMatrixIR.at<double>(0,0);
  ir_params.fy = cameraMatrixIR.at<double>(1,1);
  ir_params.cx = cameraMatrixIR.at<double>(0,2);
  ir_params.cy = cameraMatrixIR.at<double>(1,2);
  ir_params.k1 = distCoeffsIR.at<double>(0);
  ir_params.k2 = distCoeffsIR.at<double>(1);
  ir_params.p1 = distCoeffsIR.at<double>(2);
  ir_params.p2 = distCoeffsIR.at<double>(3);
  ir_params.k3 = distCoeffsIR.at<double>(4);
  dev.open(0);
  dev.setIRParams(ir_params);
  dev.setColorParams(color_params);
  //dev.open(1);
  //dev.setIRParams(ir_params);
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
  constexpr int filterTimes = 10;
  int filterCounter = 0;
  auto scenario_iter = base_scenario.end() - 1;

  std::thread gl_thread(farsight::init3d);
  gl_thread.detach();
  detector dec;
  kinect k_dev(0);
  auto irParams0 = k_dev.getIRParams();
  auto colorParams0 = k_dev.getColorParams();
  k_dev.open(1);
  auto irParams1 = k_dev.getIRParams();
  auto colorParams1 = k_dev.getColorParams();

  libfreenect2::Registration reg[2]{{irParams0, colorParams0}, {irParams1, colorParams1}};
  k_dev.open(selectedKinnect);

  shared_t shared{std::mutex(), reg[selectedKinnect]};
  cv::namedWindow(wndname2, cv::WINDOW_AUTOSIZE);
  cv::setMouseCallback(wndname2, mouse_event_handler, &shared);

  namedWindow("floor", WINDOW_AUTOSIZE); // Create Window
  createTrackbar("Floor level",
                 "floor",
                 &floor_level_raw,
                 floor_level_max,
                 on_trackbar);
  createTrackbar("Disjoint tresholds",
                 "floor",
                 &disjointTreshold,
                 100,
                 on_disjoint_treshold);
  createTrackbar("Disjoint set valid size",
                 "floor",
                 &disjointSetValidSize,
                 300,
                 on_disjoint_valid_size);

  byte *depth_backup = nullptr;
  while (continue_flag.test_and_set() and c != 'q')
  {
    k_dev.waitForFrames(10);

    rgb = k_dev.frames[libfreenect2::Frame::Color];
    ir = k_dev.frames[libfreenect2::Frame::Ir];
    depth = k_dev.frames[libfreenect2::Frame::Depth];

    if(c == 'p'){
        memcpy(depth_frame_cpy.data,
          depth->data,
          depth_width * depth_height * sizeof(float));
    }

    depth_backup = depth->data;

    if (*scenario_iter == 'b' || *scenario_iter == 'r' || *scenario_iter == 'n')
    {
      stage1.apply(*depth);
      if (filterCounter < filterTimes)
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
    auto image_rgb = cv::Mat(rgb->height, rgb->width, CV_8UC4, rgb->data);

    if (arucoCalibrated == true)
    {
      findAruco(image_rgb);
      if (c == 's')
      {
        if (tvecs.size() && rvecs.size()){
            const auto &tvec = tvecs[0];
            const auto &rvec = rvecs[0];
            farsight::Point3f pos(tvec[0], tvec[1], tvec[2]);
            farsight::Point3f rot(rvec[0], rvec[1], rvec[2]);
            dec.setCameraFaceID(selectedKinnect, ids[0]);
            dec.setCameraPos(selectedKinnect, pos);
            dec.setCameraRot(selectedKinnect, rot);
            distance = dec.calcMaxDistance();
            generateScene(reg[selectedKinnect], depth, pos, rot, selectedKinnect);
        }
      }
    }

    depthProcess(depth);
    conv32FC1To8CU1(depth->data, depth->height * depth->width);
    auto image_depth =
      cv::Mat(depth->height, depth->width, CV_8UC1, depth->data);
    switch (c)
    {
      case 'e':
        fmt::print("calibration started");
        cv::glob("../../../aruco/*.jpg", images);
        cv::glob("../../../aruco_ir/*.jpg", images_ir);
        calibrateArucoColor();
        calibrateArucoIr();
        arucoCalibrated = true;
        break;
      case 'c': {
        cv::Vec3d tvec;
        cv::Vec3d rvec;
        cv::FileStorage storage(arucoConfigPath, cv::FileStorage::WRITE);
        storage << "cameraMatrix" << cameraMatrix;
        storage << "distCoeffs" << distCoeffs;
        storage << "cameraMatrixIR" << cameraMatrixIR;
        storage << "distCoeffsIR" << distCoeffsIR;
        const auto &pos1 = dec.getCameraPos(0);
        tvec[0] = pos1.x;
        tvec[1] = pos1.y;
        tvec[2] = pos1.z;
        storage << "tvec_cam1" << tvec;
        const auto &pos2 = dec.getCameraPos(1);
        tvec[0] = pos2.x;
        tvec[1] = pos2.y;
        tvec[2] = pos2.z;
        storage << "tvec_cam2" << tvec;

        const auto &rot1 = dec.getCameraRot(0);
        rvec[0] = rot1.x;
        rvec[1] = rot1.y;
        rvec[2] = rot1.z;
        storage << "rvec_cam1" << rvec;
        const auto &rot2 = dec.getCameraRot(1);
        rvec[0] = rot2.x;
        rvec[1] = rot2.y;
        rvec[2] = rot2.z;
        storage << "rvec_cam2" << rvec;

        tvec[0] = cam1_tvec.x;
        tvec[1] = cam1_tvec.y;
        tvec[2] = cam1_tvec.z;
        storage << "glvec_cam1" << tvec;
        tvec[0] = cam2_tvec.x;
        tvec[1] = cam2_tvec.y;
        tvec[2] = cam2_tvec.z;
        storage << "glvec_cam2" << tvec;

        tvec[0] = cam1_rvec.x;
        tvec[1] = cam1_rvec.y;
        tvec[2] = cam1_rvec.z;
        storage << "glrot_cam1" << tvec;
        tvec[0] = cam2_rvec.x;
        tvec[1] = cam2_rvec.y;
        tvec[2] = cam2_rvec.z;
        storage << "glrot_cam2" << tvec;

        storage << "face_id_1" << dec.getCameraFaceID(0);
        storage << "face_id_2" << dec.getCameraFaceID(1);
        storage << "distance" << distance;
        storage << "floor_level" << floor_level;
        calibrateCamera(k_dev);
        storage.release();
        arucoCalibrated = true;
      }
      break;
      case 'l': {
        cv::Vec3d vec;
        farsight::Point3f pos;
        glm::vec3 gpos;
        int faceid;
        cv::FileStorage storage(arucoConfigPath, cv::FileStorage::READ);
        storage["cameraMatrix"] >> cameraMatrix;
        storage["distCoeffs"] >> distCoeffs;
        storage["cameraMatrixIR"] >> cameraMatrixIR;
        storage["distCoeffsIR"] >> distCoeffsIR;
        storage["tvec_cam1"] >> vec;
        pos.x = vec[0];
        pos.y = vec[1];
        pos.z = vec[2];
        dec.setCameraPos(0, pos);
        storage["tvec_cam2"] >> vec;
        pos.x = vec[0];
        pos.y = vec[1];
        pos.z = vec[2];
        dec.setCameraPos(1, pos);
        storage["rvec_cam1"] >> vec;
        pos.x = vec[0];
        pos.y = vec[1];
        pos.z = vec[2];
        dec.setCameraRot(0, pos);
        storage["rvec_cam2"] >> vec;
        pos.x = vec[0];
        pos.y = vec[1];
        pos.z = vec[2];
        dec.setCameraRot(1, pos);
        storage["glvec_cam1"] >> vec;
        cam1_tvec.x = vec[0];
        cam1_tvec.y = vec[1];
        cam1_tvec.z = vec[2];
        farsight::set_tvec_cam1(cam1_tvec);
        storage["glvec_cam2"] >> vec;
        cam2_tvec.x = vec[0];
        cam2_tvec.y = vec[1];
        cam2_tvec.z = vec[2];
        farsight::set_tvec_cam2(cam2_tvec);
        storage["glrot_cam1"] >> vec;
        cam1_rvec.x = vec[0];
        cam1_rvec.y = vec[1];
        cam1_rvec.z = vec[2];
        farsight::set_rvec_cam1(cam1_rvec);
        storage["glrot_cam2"] >> vec;
        cam2_rvec.x = vec[0];
        cam2_rvec.y = vec[1];
        cam2_rvec.z = vec[2];
        farsight::set_rvec_cam2(cam2_rvec);
        //calibrateCamera(k_dev);  
        storage["face_id_1"] >> faceid;
        dec.setCameraFaceID(0, faceid);
        storage["face_id_2"] >> faceid;
        dec.setCameraFaceID(1, faceid);
        storage["distance"] >> distance;
        storage["floor_level"] >> floor_level;
        storage.release();
        arucoCalibrated = true;
      }
      break;
      case 'x':
        cam1_tvec += farsight::get_tvec_cam1();
        cam1_rvec += farsight::get_rvec_cam1();
        cam2_tvec += farsight::get_tvec_cam2();
        cam2_rvec += farsight::get_rvec_cam2();
        farsight::set_tvec_cam1({0,0,0});
        farsight::set_tvec_cam1({0,0,0});
        farsight::set_rvec_cam2({0,0,0});
        farsight::set_rvec_cam2({0,0,0});
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

    switch (*scenario_iter)
    {
      case 'b': {
        fmt::print("Setting {} kinect base image \n", selectedKinnect + 1);
        dec.saveBaseDepthImg(selectedKinnect, image_depth);
        dec.displayCurrectConfig();
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
      case 'r': {
        auto depth_cpy = image_depth.clone();
        dec.saveDepthFrame(selectedKinnect, objectType::REFERENCE_OBJ, &depth_frame_cpy);
        const auto faceid = dec.getCameraFaceID(selectedKinnect);
        const auto &pos = dec.getCameraPos(selectedKinnect);
        const auto &rot = dec.getCameraRot(selectedKinnect);
        auto detectedBox = dec.detect(
          selectedKinnect, depth->data, total_size_depth, depth_cpy);
        const auto &np = dec.getNearestPoint(selectedKinnect == 0 ? 1 : 0);
        double dist = distance - np.z;
        fmt::print("Distance {}, nearest point {}\n", dist, np.z);
        auto realPoints = createPointMaping(reg[selectedKinnect],
                                            &depth_frame_cpy,
                                            depth->data,
                                            detectedBox,
                                            pos,
                                            rot,
                                            faceid,
                                            selectedKinnect,
                                            dist);

        dec.setConfig(
          selectedKinnect, objectType::REFERENCE_OBJ, depth_cpy, detectedBox, realPoints);
        dec.displayCurrectConfig();
        auto minRect = dec.calcBiggestComponent();
        auto mass_center = minRect.center;
        mass_center.x/=1000;
        mass_center.y/=1000;
        auto angle = minRect.angle;
        double obj_width = minRect.size.width/1000.0;
        double obj_height = minRect.size.height/1000.0;
        fmt::print("MASS CENETER {} {}\n", mass_center.x, mass_center.y);

        farsight::Rectfc corners;
        corners.verts[0] = {
          static_cast<float>(mass_center.x + obj_width / 2),
          0.0,
          static_cast<float>(mass_center.y + obj_height / 2),
          farsight::WHITE
        };
        corners.verts[1] = {
          static_cast<float>(mass_center.x + obj_width / 2),
          0.0,
          static_cast<float>(mass_center.y - obj_height / 2),
          farsight::WHITE
        };
        corners.verts[2] = {
          static_cast<float>(mass_center.x - obj_width / 2),
          0.0,
          static_cast<float>(mass_center.y - obj_height / 2),
          farsight::WHITE
        };
        corners.verts[3] = {
          static_cast<float>(mass_center.x - obj_width / 2),
          0.0,
          static_cast<float>(mass_center.y + obj_height / 2),
          farsight::WHITE
        };
        glm::vec3 rotRectMat = {0.0,(M_PI/180)*angle, 0.0};
        fmt::print("RECT CORNER_1 {} {} {}\n", corners.verts[0].x, corners.verts[0].y, corners.verts[0].z);
        fmt::print("RECT CORNER_2 {} {} {}\n", corners.verts[1].x, corners.verts[1].y, corners.verts[1].z);
        fmt::print("RECT CORNER_3 {} {} {}\n", corners.verts[2].x, corners.verts[2].y, corners.verts[2].z);
        fmt::print("RECT CORNER_4 {} {} {}\n", corners.verts[3].x, corners.verts[3].y, corners.verts[3].z);
        fmt::print("Corner angle {}\n", rotRectMat.y);
        farsight::reset_marks();
        farsight::add_marker(corners, {0,0,0}, rotRectMat);
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

    if (*scenario_iter == 'b' || *scenario_iter == 'n' ||
        *scenario_iter == 'r')
    {
      stage1.reset();
      depth->data = depth_backup;
    }

    if (*scenario_iter != 'e')
    {
      fmt::print("Scenario iter: {}\n", *scenario_iter);
      scenario_iter++;
    }

    if (c == 'b')
    {
      if (*scenario_iter == 'e')
      {
        scenario_iter = base_scenario.begin();
        std::this_thread::sleep_for(std::chrono::seconds(4));
      }
    }
    else if (c == 'm')
    {
      if (*scenario_iter == 'e')
      {
        scenario_iter = meassure_scenario.begin();
        std::this_thread::sleep_for(std::chrono::seconds(4));
      }
    }
    k_dev.releaseFrames();
  }
  k_dev.close();
}
