#pragma once
#include "image_utils.hpp"
#include "config.hpp"
#include <type_traits>
#include <memory>
#include <opencv2/core/types.hpp>
#include <libfreenect2/frame_listener.hpp>
#include "types.h"


enum class objectType : unsigned int
{
    REFERENCE_OBJ,
    MEASURED_OBJ
};
template <typename E>
constexpr auto to_underlying(E e) noexcept
{
    return static_cast<std::underlying_type_t<E>>(e);
}

struct object_t
{
    bbox area;
    farsight::Point3f nearest_point;
    cv::Mat imgDepth = cv::Mat::zeros(
        cv::Size(depth_width, depth_height), CV_8UC1); 
    libfreenect2::Frame depthFrame =
      libfreenect2::Frame(depth_width, depth_height, sizeof(float));
    farsight::PointArray pointCloud;
    bool configured = false;
};

constexpr int objectsPerCamera = 2;
using objectArray = std::array<object_t, objectsPerCamera>;

struct cameraConfig
{
    farsight::Point3f camPose {0,0,0};
    farsight::Point3f camRot {0,0,0};
    cv::Mat img_base = cv::Mat::zeros(
        cv::Size(depth_width, depth_height), CV_8UC1);;
    libfreenect2::Frame base =
      libfreenect2::Frame(depth_width, depth_height, sizeof(float));
    objectArray objects;
    int camSpan;
};
