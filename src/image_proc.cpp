#include "image_proc.hpp"
#include <fmt/format.h>

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
bbox
detector::detect(int kinectID,
                 const byte *frameObject,
                 size_t size,
                 cv::Mat &imageDepth)
{
  int lowerb = 0, higherb = 255;
  int lowerb2 = 20, higherb2 = 240;

  std::array<byte, depth_width * depth_height> frame_depth_;
  std::copy(frameObject,
            frameObject + depth_width * depth_height,
            frame_depth_.data());

  diff(frame_depth_.data(),
       config[kinectID].img_base.ptr(),
       size);
  auto image_depth_ =
    cv::Mat(depth_height, depth_width, CV_8UC1, frame_depth_.data());
  cv::Mat image_depth_th, image_depth_filtered;
  cv::threshold(
    image_depth_, image_depth_th, lowerb, higherb, cv::THRESH_BINARY_INV);
  cv::inRange(image_depth_, lowerb2, higherb2, image_depth_filtered);

  cv::Mat labels, stats, centroids;
  cv::connectedComponentsWithStats(
    image_depth_filtered, labels, stats, centroids);

  bbox best_bbox;

  for (int i = 0; i < stats.rows; ++i)
  {
    int x = stats.at<int>({ 0, i });
    int y = stats.at<int>({ 1, i });
    int w = stats.at<int>({ 2, i });
    int h = stats.at<int>({ 3, i });
    int area = stats.at<int>({ 4, i });

    if (area >= depth_width * depth_height / 2)
      continue;

    if (area > best_bbox.area)
    {
      best_bbox.area = area;
      best_bbox.x = x;
      best_bbox.y = y;
      best_bbox.w = w;
      best_bbox.h = h;
    }
  }

  fmt::print("x : {}, y {}, w : {}, h {},  Area: {}\n",
             best_bbox.x,
             best_bbox.y,
             best_bbox.w,
             best_bbox.h,
             best_bbox.area);
  cv::Scalar color(0, 0, 0);
  cv::Rect rect(best_bbox.x, best_bbox.y, best_bbox.w, best_bbox.h);
  cv::rectangle(imageDepth, rect, color, 3);
  return best_bbox;
}

void
detector::setConfig(int kinectID,
                    objectType t,
                    const cv::Mat &img,
                    const bbox &a,
                    const bbox &ra,
                    const position &p,
                    const pointArray &flattened)
{
  auto &c =config[kinectID].objects[to_underlying(t)];
  c.area = a;
  c.realArea = ra;
  c.nearest_point = p;
  img.copyTo(c.imgDepth);
  c.flattenedObject = flattened;
  c.configured = true;
}

void
detector::calcReferenceOffsset(objectType t)
{
  auto &c1 =config[0].objects[to_underlying(t)];
  auto &c2 =config[1].objects[to_underlying(t)];
  if(!(c1.configured && c2.configured))
    return;

  int span_x = (double(c1.realArea.w)/c1.area.w)*depth_width;
  int off_1, off_2;
  off_1 = c1.realArea.x;
  // since second camer has reflected image of first camera
  // we need to take second vertex and substract it from max width
  off_2 = span_x - (c2.realArea.x + c2.realArea.w);
  cameraOffsets.x = off_1 - off_2; 
  cameraOffsets.y = c1.realArea.y - c2.realArea.y;
  cameraOffsets.z = c1.nearest_point.z + c2.nearest_point.z + 500;
}

void
detector::calcBiggestComponent()
{
  auto &c1 =config[0].objects[to_underlying(objectType::MEASURED_OBJ)];
  auto &c2 =config[1].objects[to_underlying(objectType::MEASURED_OBJ)];
  if(!(c1.configured && c2.configured))
    return;

  pointArray pointsCloud(c1.flattenedObject); 
  // translate second camera points based on reference objects 
  for(const auto &p : c2.flattenedObject)
  {
    pointsCloud.emplace_back(p.x-cameraOffsets.x, p.y); // need to do sth wit z axies
  }

  cv::minAreaRect(pointsCloud);
}

void
detector::displayCurrectConfig()
{
  const auto &c1_r = config[0].objects[to_underlying(objectType::REFERENCE_OBJ)];
  const auto &c1_m = config[0].objects[to_underlying(objectType::MEASURED_OBJ)];
  const auto &c2_r = config[1].objects[to_underlying(objectType::REFERENCE_OBJ)];
  const auto &c2_m = config[1].objects[to_underlying(objectType::MEASURED_OBJ)];
  cv::Mat temp;

  matRoi = cv::Rect(0, 0, depth_width, depth_height);
  resize(c1_r.imgDepth, temp, cv::Size(depth_width, depth_height));
  cv::putText(temp,
              "c1 base",
              { 50, 50 },
              cv::FONT_HERSHEY_PLAIN,
              2,
              cv::Scalar::all(255),
              3,
              5);
  temp.copyTo(configScreen(matRoi));

  matRoi = cv::Rect(depth_width, 0, depth_width, depth_height);
  resize(c1_m.imgDepth, temp, cv::Size(depth_width, depth_height));
  cv::putText(temp,
              fmt::format("object {}x{} pixels ", c1_m.area.w, c1_m.area.h),
              { depth_width / 10, 50 },
              cv::FONT_HERSHEY_PLAIN,
              2,
              cv::Scalar::all(255),
              3,
              5);
  cv::putText(temp,
              fmt::format("object depth {} cm ", c2_r.nearest_point.z),
              { depth_width / 10, 100 },
              cv::FONT_HERSHEY_PLAIN,
              2,
              cv::Scalar::all(255),
              3,
              5);
  temp.copyTo(configScreen(matRoi));

  matRoi = cv::Rect(0, depth_height, depth_width, depth_height);
  resize(c2_r.imgDepth, temp, cv::Size(depth_width, depth_height));
  cv::putText(temp,
              "c2 base",
              { 50, depth_height / 10 },
              cv::FONT_HERSHEY_PLAIN,
              2,
              cv::Scalar::all(255),
              3,
              5);
  temp.copyTo(configScreen(matRoi));

  matRoi =
    cv::Rect(depth_width, depth_height, depth_width, depth_height);
  resize(c2_m.imgDepth, temp, cv::Size(depth_width, depth_height));
  cv::putText(temp,
              fmt::format("object {}x{} pixels ", c2_m.area.w, c2_m.area.h),
              { depth_width / 10, depth_height / 10 },
              cv::FONT_HERSHEY_PLAIN,
              2,
              cv::Scalar::all(255),
              3,
              5);
  cv::putText(temp,
              fmt::format("object depth {} cm ", c2_m.nearest_point.z),
              { depth_width / 10, 100 },
              cv::FONT_HERSHEY_PLAIN,
              2,
              cv::Scalar::all(255),
              3,
              5);
  temp.copyTo(configScreen(matRoi));

  cv::imshow("config", configScreen);
}

void
detector::saveDepthFrame(int kinectID,
                         const objectType t,
                         const libfreenect2::Frame *frame)
{
  auto &c = config[kinectID].objects[to_underlying(t)];
  std::copy(frame->data,
            frame->data + total_size_depth*sizeof(float),
            c.depthFrame.get());
}

void
detector::setBaseImg(int kinectID, const cv::Mat &img)
{
  auto &c = config[kinectID];
  img.copyTo(c.img_base);
}

void
detector::meassure()
{}
