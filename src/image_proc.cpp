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
                 const byte *frame_object,
                 size_t size,
                 cv::Mat &image_depth)
{
  int lowerb = 0, higherb = 255;
  int lowerb2 = 20, higherb2 = 240;
  bool clr = false;

  std::array<byte, depth_width * depth_height> frame_depth_;
  std::copy(frame_object,
            frame_object + depth_width * depth_height,
            frame_depth_.data());

  diff(frame_depth_.data(),
       objectConfigs[kinectID].img_base.ptr(),
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

  clr = !clr;
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
  cv::Scalar color(clr ? 255 : 0, 0, 0);
  cv::Rect rect(best_bbox.x, best_bbox.y, best_bbox.w, best_bbox.h);
  cv::rectangle(image_depth, rect, color, 3);
  return best_bbox;
}

void
detector::configure(int kinectID,
                    const cv::Mat &img,
                    const bbox &sizes,
                    const depth_t &dep)
{
  auto &c = objectConfigs[kinectID];
  img.copyTo(c.img_object);
  c.area = sizes;
  c.nearest_point = dep;
  c.imObjectSet = true;
}

void
detector::setBaseImg(int kinectID, const cv::Mat &img)
{
  auto &c = objectConfigs[kinectID];
  img.copyTo(c.img_base);
  img.copyTo(c.img_base);
  c.imBaseSet = true;
}

void
detector::displayCurrectConfig()
{
  const auto &c1 = objectConfigs[0];
  const auto &c2 = objectConfigs[1];
  cv::Mat temp;

  if (c1.imBaseSet == true)
  {
    matRoi = cv::Rect(0, 0, depth_width, depth_height);
    resize(c1.img_base, temp, cv::Size(depth_width, depth_height));
    cv::putText(temp,
                "c1 base",
                { 50, 50 },
                cv::FONT_HERSHEY_PLAIN,
                2,
                cv::Scalar::all(255),
                3,
                5);
    temp.copyTo(configScreen(matRoi));
  }

  if (c1.imObjectSet == true)
  {
    matRoi = cv::Rect(depth_width, 0, depth_width, depth_height);
    resize(c1.img_object, temp, cv::Size(depth_width, depth_height));
    cv::putText(temp,
                fmt::format("object {}x{} pixels ", c1.area.w, c1.area.h),
                { depth_width / 10, 50 },
                cv::FONT_HERSHEY_PLAIN,
                2,
                cv::Scalar::all(255),
                3,
                5);
    cv::putText(temp,
                fmt::format("object depth {} cm ", c1.nearest_point.depth),
                { depth_width / 10, 100 },
                cv::FONT_HERSHEY_PLAIN,
                2,
                cv::Scalar::all(255),
                3,
                5);
    temp.copyTo(configScreen(matRoi));
  }

  if (c2.imBaseSet == true)
  {
    matRoi = cv::Rect(0, depth_height, depth_width, depth_height);
    resize(c2.img_base, temp, cv::Size(depth_width, depth_height));
    cv::putText(temp,
                "c2 base",
                { 50, depth_height / 10 },
                cv::FONT_HERSHEY_PLAIN,
                2,
                cv::Scalar::all(255),
                3,
                5);
    temp.copyTo(configScreen(matRoi));
  }

  if (c2.imObjectSet == true)
  {
    matRoi =
      cv::Rect(depth_width, depth_height, depth_width, depth_height);
    resize(c2.img_object, temp, cv::Size(depth_width, depth_height));
    cv::putText(temp,
                fmt::format("object {}x{} pixels ", c2.area.w, c2.area.h),
                { depth_width / 10, depth_height / 10 },
                cv::FONT_HERSHEY_PLAIN,
                2,
                cv::Scalar::all(255),
                3,
                5);
    cv::putText(temp,
                fmt::format("object depth {} cm ", c2.nearest_point.depth),
                { depth_width / 10, 100 },
                cv::FONT_HERSHEY_PLAIN,
                2,
                cv::Scalar::all(255),
                3,
                5);
    temp.copyTo(configScreen(matRoi));
  }
  cv::imshow("config", configScreen);
}

void
detector::saveOriginalFrameObject(int kinectID,
                                  const libfreenect2::Frame *frame)
{
  auto &c = objectConfigs[kinectID];
  std::copy(frame->data,
            frame->data + detector::depth_total_size,
            c.originalObjectFrame.get());
}

void
detector::meassure()
{}

bool
detector::isFullyConfigured()
{
  const auto &c1 = objectConfigs[0];
  const auto &c2 = objectConfigs[1];

  return c1.imBaseSet && c1.imObjectSet && c2.imBaseSet && c2.imObjectSet;
}
