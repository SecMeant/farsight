#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/photo.hpp>

#include "kinect_manager.hpp"

#include <fmt/format.h>
#include <iostream>
#include <stdio.h>

#include <string>

const cv::Ptr<cv::aruco::Dictionary> dict =
  cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
const cv::Ptr<cv::aruco::CharucoBoard> chboard =
  cv::aruco::CharucoBoard::create(5, 7, 0.035f, 0.021f, dict);
const cv::Size img_size(1080, 1920);

void
depthProcess(libfreenect2::Frame *frame)
{
  auto total_size = frame->height * frame->width;
  auto fp = reinterpret_cast<float *>(frame->data);

  for (int i = 0; i < total_size; i++)
  {
    fp[i] /= 65535.0f;
  }
}

void
conv32FC1To8CU1(unsigned char *data, size_t size)
{
  auto fp = reinterpret_cast<float *>(data);

  for (auto i = 0; i < size; ++i, ++fp, ++data)
    *data = static_cast<unsigned char>(*fp * 255.0f);
}

enum class FrameGrabberType
{
  RGB,
  IR,
  UNKNOWN
};

struct FrameGrabber
{
  FrameGrabber(FrameGrabberType type)
    : type(type)
  {}

  cv::Mat
  grab(kinect &kdev)
  {
    cv::Mat image;

    switch (this->type)
    {
      case FrameGrabberType::RGB: {
        libfreenect2::Frame *rgb = kdev.frames[libfreenect2::Frame::Color];
        image = cv::Mat(rgb->height, rgb->width, CV_8UC4, rgb->data);
        cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
        break;
      }

      case FrameGrabberType::IR: {
        libfreenect2::Frame *ir = kdev.frames[libfreenect2::Frame::Ir];

        depthProcess(ir);
        conv32FC1To8CU1(ir->data, ir->height * ir->width);

        image = cv::Mat(ir->height, ir->width, CV_8UC1, ir->data);
        cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);

        break;
      }
      case FrameGrabberType::UNKNOWN: {
        assert(false);
      }
    }

    cv::flip(image, image, 1);
    return image;
  }

  const FrameGrabberType type;
};

static auto
charuco_find_precalib(kinect &kdev, FrameGrabber fg)
{
  cv::Ptr<cv::aruco::DetectorParameters> params =
    cv::aruco::DetectorParameters::create();
  std::vector<std::vector<cv::Point2f>> all_charuco_corners;
  std::vector<std::vector<int>> all_charuco_ids;

  bool save_next = false;

  while (1)
  {
    kdev.waitForFrames(10);

    cv::Mat image = fg.grab(kdev), image_copy;

    image.copyTo(image_copy);
    std::vector<int> marker_ids;
    std::vector<std::vector<cv::Point2f>> marker_corners;
    cv::aruco::detectMarkers(
      image, chboard->dictionary, marker_corners, marker_ids, params);

    if (marker_ids.size() > 0)
    {
      cv::aruco::drawDetectedMarkers(
        image_copy, marker_corners, marker_ids);
      std::vector<cv::Point2f> charuco_corners;
      std::vector<int> charuco_ids;
      cv::aruco::interpolateCornersCharuco(marker_corners,
                                           marker_ids,
                                           image,
                                           chboard,
                                           charuco_corners,
                                           charuco_ids);

      if (charuco_ids.size() > 0)
      {
        cv::aruco::drawDetectedCornersCharuco(
          image_copy, charuco_corners, charuco_ids, cv::Scalar(255, 0, 0));
        if (save_next)
        {
          save_next = false;
          all_charuco_ids.emplace_back(std::move(charuco_ids));
          all_charuco_corners.emplace_back(std::move(charuco_corners));

          fmt::print("Saved charuco frame\n");
        }
      }
    }

    cv::imshow("out", image_copy);
    char key = (char)cv::waitKey(30);

    if (key == 'q')
      break;
    else if (key == ' ')
      save_next = true;

    kdev.releaseFrames();
  }

  return std::make_tuple(std::move(all_charuco_corners),
                         std::move(all_charuco_ids));
}

constexpr auto settings_filename = "charuco_settings.yaml";

static void
save_settings(cv::Mat &camera_matrix, cv::Mat &dist_coeffs)
{
  cv::FileStorage fs(settings_filename, cv::FileStorage::WRITE);

  if (!fs.isOpened())
  {
    fmt::print("Warning failed to save settings.\n");
    return;
  }

  fs << "camera_matrix" << camera_matrix;
  fs << "dist_coeffs" << dist_coeffs;
}

static void
load_settings(cv::Mat &camera_matrix, cv::Mat &dist_coeffs)
{
  cv::FileStorage fs(settings_filename, cv::FileStorage::READ);

  if (!fs.isOpened())
  {
    fmt::print("Warning failed to load settings.\n");
    return;
  }

  fs["camera_matrix"] >> camera_matrix;
  fs["dist_coeffs"] >> dist_coeffs;
}

int
main(int argc, char **argv)
{
  FrameGrabberType image_type;
  if (argc == 1)
  {
    image_type = FrameGrabberType::RGB;
  }
  else if (argc == 2)
  {
    std::string type_str = argv[1];
    std::for_each(std::begin(type_str), std::end(type_str), [](char &c) {
      c = std::toupper(c);
    });

    if (type_str == "RGB")
    {
      image_type = FrameGrabberType::RGB;
    }
    else if (type_str == "IR")
    {
      image_type = FrameGrabberType::IR;
    }
    else
    {
      puts("Unknown image type. Try RGB or IR\n");
      return 1;
    }
  }

  FrameGrabber grabber(image_type);

  kinect kdev(0);

  auto [charuco_corners, charuco_ids] =
    charuco_find_precalib(kdev, grabber);

  cv::Mat camera_matrix, dist_coeffs;
  std::vector<cv::Mat> rvecs, tvecs;

  cv::aruco::calibrateCameraCharuco(charuco_corners,
                                    charuco_ids,
                                    chboard,
                                    img_size,
                                    camera_matrix,
                                    dist_coeffs,
                                    rvecs,
                                    tvecs,
                                    0);

  save_settings(camera_matrix, dist_coeffs);
}
