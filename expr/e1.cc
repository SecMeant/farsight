#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

#include <memory>
#include <stdio.h>

#include <cmath>

#include <fmt/format.h>
#include <fmt/ostream.h>

#include "trans.cc"
const char *wndname = "wnd";
const char *wndname2 = "wnd2";
const char *wndname3 = "wnd3";
const char *wndname4 = "wnd4";

using byte = unsigned char;

void
gamma(float *data, size_t size, float gamma)
{
  for (auto i = 0; i < size; ++i)
    data[i] = std::pow(data[i], 1.0f / gamma);
}

constexpr size_t cubeface_count = 6;
std::array<cv::Mat, cubeface_count> cubefaces;

bool
loadfaces()
{
  cubefaces[0] = cv::imread("media/cube_photo0.jpg", cv::IMREAD_GRAYSCALE);
  cubefaces[1] = cv::imread("media/cube_photo1.jpg", cv::IMREAD_GRAYSCALE);
  cubefaces[2] = cv::imread("media/cube_photo2.jpg", cv::IMREAD_GRAYSCALE);
  cubefaces[3] = cv::imread("media/cube_photo3.jpg", cv::IMREAD_GRAYSCALE);
  cubefaces[4] = cv::imread("media/cube_photo4.jpg", cv::IMREAD_GRAYSCALE);
  cubefaces[5] = cv::imread("media/cube_photo5.jpg", cv::IMREAD_GRAYSCALE);

  for (auto i = 0; i < cubeface_count; ++i)
    if (!cubefaces[i].data)
      return false;

  return true;
}

void
removeAlpha(float *data, size_t size)
{
  assert(size % 4 == 0);

  for (auto current = 0, insert = 0; current < size;
       current += 4, insert += 3)
  {
    data[insert + 0] = data[current + 0];
    data[insert + 1] = data[current + 1];
    data[insert + 2] = data[current + 2];
  }
}

void
conv8UC4To32FC4(byte *data, size_t size)
{
  auto rgbend = data + size;
  auto rgbp = rgbend - (size / sizeof(float));

  for (auto rgbfp = reinterpret_cast<float *>(data); rgbp != rgbend;
       ++rgbp, ++rgbfp)
  {
    *rgbfp = static_cast<float>(*rgbp) / 255.0f;
  }
}

void diff(byte *i1_, byte *i2_, size_t size, char th = 10)
{
  auto i1 = reinterpret_cast<char*>(i1_);
  auto i2 = reinterpret_cast<char*>(i2_);

  for (auto i = 0; i < size; ++i)
  {
    if (std::abs(i1[i] - i2[i]) < th)
      i1[i] = 255;
  }
}

void conv8UC4To32FC1(byte *data, size_t size)
{
  auto rgbend = data + size;
  auto rgbp = rgbend - (size / sizeof(float));

  for (auto rgbfp = reinterpret_cast<float *>(data); rgbp != rgbend;
       rgbp+=4, ++rgbfp)
  {
    auto b = static_cast<float>(rgbp[0]) / 255.0f;
    auto g = static_cast<float>(rgbp[1]) / 255.0f;
    auto r = static_cast<float>(rgbp[2]) / 255.0f;

    *rgbfp = (b + g + r ) / 3.0f;
  }
}

void conv32FC1To8CU1(byte *data, size_t size)
{
  auto fp = reinterpret_cast<float*>(data);

  for (auto i = 0; i < size; ++i, ++fp, ++data)
    *data = static_cast<byte>(*fp * 255.0f);
}

constexpr auto bfm_ctype = CV_8UC1;

int
main(int argc, char **argv)
{
  size_t depth_width = 512, depth_height = 424;
  size_t fsize_depth = depth_width * depth_height;
  size_t total_size_depth = fsize_depth * sizeof(float);

  size_t rgb_width = 1920, rgb_height = 1080;
  size_t fsize_rgb = rgb_width * rgb_height;
  size_t total_size_rgb = fsize_rgb * sizeof(float) * 4;

  auto imgraw_depth_base_ = std::make_unique<byte[]>(total_size_depth);
  auto imgraw_depth_base = std::make_unique<byte[]>(total_size_depth);

  auto imgraw_depth_ = std::make_unique<byte[]>(total_size_depth);
  auto imgraw_depth = std::make_unique<byte[]>(total_size_depth);

  auto imgraw_rgb_ = std::make_unique<byte[]>(total_size_rgb);
  auto imgraw_rgb = std::make_unique<byte[]>(total_size_rgb);

  auto imgraw_rgb_base_ = std::make_unique<byte[]>(total_size_rgb);
  auto imgraw_rgb_base = std::make_unique<byte[]>(total_size_rgb);

  auto fno = argc >= 2 ? atoi(argv[1]) : 2;

  auto f = fopen(fmt::format("media/depth_raw{}", fno).c_str(), "rb");

  if (!f)
    return 1;

  if (fread(imgraw_depth_.get(), 4, depth_width * depth_height, f) !=
      depth_width * depth_height)
    return -1;
  fclose(f);

  f = fopen(fmt::format("media/depth_raw{}", fno > 5 ? 5 : 0).c_str(), "rb");

  if (!f)
    return 1;

  if (fread(imgraw_depth_base_.get(), 4, depth_width * depth_height, f) !=
      depth_width * depth_height)
    return -1;
  fclose(f);

  auto rgbend = imgraw_rgb_.get() + total_size_rgb;
  auto rgbp = rgbend - (total_size_rgb / sizeof(float));

  f = fopen(fmt::format("media/rgb_raw{}", fno).c_str(), "rb");

  if (!f)
    return 2;

  if (fread(imgraw_rgb_.get(), 4, rgb_width * rgb_height, f) != rgb_width * rgb_height)
    return -2;
  fclose(f);

  rgbend = imgraw_rgb_base_.get() + total_size_rgb;
  rgbp = rgbend - (total_size_rgb / sizeof(float));

  f = fopen(fmt::format("media/rgb_raw5", fno).c_str(), "rb");

  if (!f)
    return 2;

  if (fread(imgraw_rgb_base_.get(), 4, rgb_width * rgb_height, f) != rgb_width * rgb_height)
    return -2;
  fclose(f);


  conv32FC1To8CU1(imgraw_depth_base_.get(), depth_height * depth_width);
  conv32FC1To8CU1(imgraw_depth_.get(), depth_height * depth_width);
  //removeAlpha(reinterpret_cast<float *>(imgraw_rgb_.get()),
  //          rgb_width * rgb_height * sizeof(float));


  cv::SimpleBlobDetector::Params params;

  params.filterByArea = true;
  params.minArea = 5;
  params.maxArea = std::numeric_limits<float>::infinity();

  cv::Ptr<cv::SimpleBlobDetector> det = cv::SimpleBlobDetector::create(params);

  int c = 0;
  float g = 0.5f;

  int connectivity = 4, itype = CV_16U, ccltype = cv::CCL_WU;
  int lowerb = 0, higherb = 255;
  int lowerb2 = 20, higherb2 = 240;
  int area = 0;
  bool clr = false;
  int p = 0;
  //cv::namedWindow(wndname, cv::WINDOW_AUTOSIZE);
  //cv::createTrackbar("lowerb", wndname, &lowerb, 255);
  //cv::createTrackbar("higherb", wndname, &higherb, 255);
  //cv::createTrackbar("lowerb2", wndname, &lowerb2, 255);
  //cv::createTrackbar("higherb2", wndname, &higherb2, 255);
  //cv::createTrackbar("print", wndname, &p, 1);
  while (c != 'q')
  {
    if (c == '+')
      g += 0.015625f;
    else
      g -= 0.015625f;

    std::copy(imgraw_depth_base_.get(),
              imgraw_depth_base_.get() + total_size_depth,
              imgraw_depth_base.get());

    std::copy(imgraw_depth_.get(),
              imgraw_depth_.get() + total_size_depth,
              imgraw_depth.get());

    std::copy(imgraw_rgb_.get(),
              imgraw_rgb_.get() + total_size_rgb,
              imgraw_rgb.get());

    std::copy(imgraw_rgb_base_.get(),
              imgraw_rgb_base_.get() + total_size_rgb,
              imgraw_rgb_base.get());

    //gamma(imgf_depth, fsize_depth, g);
    diff(imgraw_depth.get(), imgraw_depth_base.get(), fsize_depth);

    auto image_depth_base =
      cv::Mat(depth_height, depth_width, CV_8UC1, imgraw_depth_base.get());
    auto image_depth =
      cv::Mat(depth_height, depth_width, CV_8UC1, imgraw_depth.get());
    auto image_rgb = cv::Mat(rgb_height, rgb_width, CV_8UC4, imgraw_rgb.get());
    auto image_rgb_base = cv::Mat(rgb_height, rgb_width, CV_8UC4, imgraw_rgb_base.get());
    auto image_th = cv::Mat(depth_height, depth_width, CV_8UC1);


    cv::Mat image_depth_th, image_depth_filtered;
    cv::threshold(image_depth, image_depth_th, lowerb, higherb, cv::THRESH_BINARY_INV);
    cv::inRange(image_depth, lowerb2, higherb2, image_depth_filtered);

    cv::Mat labels, stats, centroids;
    cv::connectedComponentsWithStats(image_depth_filtered, labels, stats, centroids);

    clr = !clr;
    bbox best_bbox = {0,0,0,0,0};

    int barea = 0;
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

    fmt::print("Area: {}\n", best_bbox.area);
    cv::Scalar color(clr ? 255 : 0, 0, 0);
    cv::Rect rect(best_bbox.x,best_bbox.y,best_bbox.w,best_bbox.h);
    cv::rectangle(image_depth, rect, color, 3);
    //std::vector<cv::KeyPoint> kp;
    //det->detect(image_depth, kp);

    //for (auto &k : kp)
    //  fmt::print("{} ", k.pt);
    //puts("");

    //cv::drawKeypoints(image_depth_th, kp, image_depth_th, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    p = 0;
    float t_x, t_y, real_x, real_y;
    bbox c_bbox;
    depth_to_color(best_bbox.x, best_bbox.y, t_x, t_y);
    f_apply(best_bbox.x, best_bbox.y, 1600, real_x,real_y, t_x, t_y);
    fmt::print("MAPPING {} {} {} {} \n", best_bbox.x, best_bbox.y, real_x, real_y);
    c_bbox.x = real_x;
    c_bbox.y = real_y;
    depth_to_color(best_bbox.x + best_bbox.w, best_bbox.y + best_bbox.h, t_x, t_y);
    f_apply(best_bbox.x, best_bbox.y, 1650, real_x,real_y, t_x, t_y);
    c_bbox.w = real_x - c_bbox.x;
    c_bbox.h = real_y - c_bbox.y;
    fmt::print("MAPPING {} {} {} {} \n", best_bbox.x + best_bbox.w, best_bbox.y + best_bbox.h, t_x, t_y);
    
    rect = cv::Rect(c_bbox.x,c_bbox.y,c_bbox.w,c_bbox.h);
    cv::rectangle(image_rgb, rect, color, 3);
    cv::imshow(wndname3, image_depth);
    cv::imshow(wndname2, image_rgb);

    c = cv::waitKey(1000);
  }
}
