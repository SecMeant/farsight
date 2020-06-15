#include "image_proc.hpp"
#include "3d.h"
#include <fmt/format.h>
#include <array>

detector::detector()
{
  cv::SimpleBlobDetector::Params params;
  params.filterByArea = true;
  params.minArea = 5;
  params.maxArea = std::numeric_limits<float>::infinity();

  det = cv::SimpleBlobDetector::create(params);
  configScreen = cv::Mat::zeros(
    cv::Size(depth_width * 2 + 10, depth_height * 2 + 10), CV_8UC1);
}

bbox detector::detect(int kinectID, byte *frame_object, size_t size, cv::Mat &image_depth)
{
  int lowerb = 0, higherb = 255;
  int lowerb2 = 20, higherb2 = 240;
  diff(frame_object, config[kinectID].img_base.ptr(), size);
  auto image_depth_ =
    cv::Mat(depth_height, depth_width, CV_8UC1, frame_object);
  cv::Mat image_depth_th, image_depth_filtered;
  cv::threshold(image_depth_, image_depth_th, lowerb, higherb, cv::THRESH_BINARY_INV);
  cv::inRange(image_depth_, lowerb2, higherb2, image_depth_filtered);
  
  cv::Mat labels, stats, centroids;
  cv::connectedComponentsWithStats(image_depth_filtered, labels, stats, centroids);
  
  bbox best_bbox;

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
  
  cv::Scalar color(255);
  cv::Rect rect(best_bbox.x,best_bbox.y,best_bbox.w,best_bbox.h);
  fmt::print("Found bbox: {} {} {} {} \n", best_bbox.x,best_bbox.y,best_bbox.w,best_bbox.h);
  cv::rectangle(image_depth, rect, color, 3);
  return best_bbox;
}

void
detector::setConfig(int kinectID,
                    objectType t,
                    const cv::Mat &img,
                    const bbox &a,
                    const farsight::PointArray &pointCloud)
{
  auto &c =config[kinectID].objects[to_underlying(t)];
  c.area = a;
  img.copyTo(c.imgDepth);
  c.pointCloud = pointCloud;
  c.configured = true;
}

cv::RotatedRect
detector::calcBiggestComponent()
{
  auto &c1 = config[0].objects[to_underlying(objectType::REFERENCE_OBJ)];
  auto &c2 = config[1].objects[to_underlying(objectType::REFERENCE_OBJ)];
  
  if(!c1.configured || !c2.configured)
    return{};
  farsight::PointArray map_;
  std::vector<cv::Point2f> pointsCloudTop;
  std::vector<cv::Point2f> pointsCloudFront;

  std::copy(
    c1.pointCloud.begin(), c1.pointCloud.end(), std::back_inserter(map_));
  FILE *file = fopen("point_cloud_0", "w");
  for(auto &p :map_)
  {
    if(std::isnan(p.x))
        continue;
    fmt::print(file, "{} {} {}\n", p.x*1000, p.y*1000, p.z*1000);
    pointsCloudFront.emplace_back(p.x*1000, p.y*1000);
    pointsCloudTop.emplace_back(p.x*1000, p.z*1000);
  }
  fclose(file);
  map_.clear();
  std::copy(
    c2.pointCloud.begin(), c2.pointCloud.end(), std::back_inserter(map_));

  FILE *file2 = fopen("point_cloud_1", "w");
  for(auto &p :map_)
  {
    if(std::isnan(p.x))
        continue;
    fmt::print(file2, "{} {} {}\n", p.x*1000, p.y*1000, p.z*1000);
    pointsCloudFront.emplace_back(p.x*1000, p.y*1000);
    pointsCloudTop.emplace_back(p.x*1000, p.z*1000);
  }

  fclose(file2);

  map_.clear();
  std::copy(
    c1.pointCloud.begin(), c1.pointCloud.end(), std::back_inserter(map_));
  std::copy(
    c2.pointCloud.begin(), c2.pointCloud.end(), std::back_inserter(map_));

  FILE *file3 = fopen("point_cloud_2", "w");
  double obj_height= 0;
  for(auto &p :map_)
  {
    if(std::isnan(p.x))
        continue;
    fmt::print(file2, "{} {} {}\n", p.x*1000, p.y*1000, p.z*1000);
    if(p.y > obj_height)
        obj_height = p.y;
    pointsCloudTop.emplace_back(p.x*1000, p.z*1000);
  }
  fclose(file3);

  if(pointsCloudTop.size() == 0 || pointsCloudFront.size() == 0)
  {
      fprintf(stderr, "No points found");
      return{};
  }

  auto rectTop = cv::minAreaRect(pointsCloudTop);
  auto obj_width = rectTop.size.width;
  auto obj_length = rectTop.size.height;


  fmt::print("Found object size : x:{} y:{} z:{}\n", obj_width, (obj_height*1000)+250, obj_length);
  return rectTop;
}

void
detector::displayCurrectConfig()
{
  const auto &cam1 = config[0];
  const auto &cam2 = config[1];

  const auto &c1_r = cam1.objects[to_underlying(objectType::REFERENCE_OBJ)];
  const auto &c1_m = cam1.objects[to_underlying(objectType::MEASURED_OBJ)];
  const auto &c2_r = cam2.objects[to_underlying(objectType::REFERENCE_OBJ)];
  const auto &c2_m = cam2.objects[to_underlying(objectType::MEASURED_OBJ)];
  cv::Mat temp;

  matRoi = cv::Rect(0, 0, depth_width, depth_height);
  resize(cam1.img_base, temp, cv::Size(depth_width, depth_height));
  temp.copyTo(configScreen(matRoi));

  matRoi = cv::Rect(depth_width, 0, depth_width, depth_height);
  resize(c1_r.imgDepth, temp, cv::Size(depth_width, depth_height));
  cv::putText(temp,
              fmt::format("object {}x{} pixels ", c1_r.area.w, c1_r.area.h),
              { depth_width / 10, 50 },
              cv::FONT_HERSHEY_PLAIN,
              2,
              cv::Scalar::all(0),
              3,
              5);
  temp.copyTo(configScreen(matRoi));

  matRoi = cv::Rect(0, depth_height, depth_width, depth_height);
  resize(cam2.img_base, temp, cv::Size(depth_width, depth_height));
  temp.copyTo(configScreen(matRoi));

  matRoi =
    cv::Rect(depth_width, depth_height, depth_width, depth_height);
  resize(c2_r.imgDepth, temp, cv::Size(depth_width, depth_height));
  cv::putText(temp,
              fmt::format("object {}x{} pixels ", c2_r.area.w, c2_r.area.h),
              { depth_width / 10, depth_height / 10 },
              cv::FONT_HERSHEY_PLAIN,
              2,
              cv::Scalar::all(0),
              3,
              5);
  temp.copyTo(configScreen(matRoi));

  cv::imshow("config", configScreen);
}
