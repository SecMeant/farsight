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

const cv::Ptr<cv::aruco::Dictionary> dict =
  cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
const cv::Ptr<cv::aruco::CharucoBoard> chboard =
  cv::aruco::CharucoBoard::create(5, 7, 0.035f, 0.021f, dict);
const cv::Size img_size(1080, 1920);

static auto
charuco_find_precalib(kinect &kdev)
{
  cv::Ptr<cv::aruco::DetectorParameters> params =
    cv::aruco::DetectorParameters::create();
  std::vector<std::vector<cv::Point2f>> all_charuco_corners;
  std::vector<std::vector<int>> all_charuco_ids;

  bool save_next = false;

  while (1)
  {
    kdev.waitForFrames(10);
    libfreenect2::Frame *rgb= kdev.frames[libfreenect2::Frame::Color];

    cv::Mat image = cv::Mat(rgb->height, rgb->width, CV_8UC4, rgb->data),
            image_copy;

    cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
    cv::flip(image, image, 1);
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

  if (!fs.isOpened()) {
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

  if (!fs.isOpened()) {
    fmt::print("Warning failed to load settings.\n");
    return;
  }

  fs["camera_matrix"] >> camera_matrix;
  fs["dist_coeffs"] >> dist_coeffs;
}

int
main()
{
  kinect kdev(0);
  auto [charuco_corners, charuco_ids] = charuco_find_precalib(kdev);

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
