#pragma once
#include "image_utils.hpp"
#include "config.hpp"
#include <type_traits>
#include <opencv2/core/types.hpp>
using pointArray = std::vector<cv::Point3f>;

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
    cv::Point3f nearest_point;
    cv::Mat imgDepth = cv::Mat::zeros(
        cv::Size(depth_width, depth_height), CV_8UC1); 
    std::unique_ptr<byte[]> depthFrame =
      std::make_unique<byte[]>(total_size_depth);
    pointArray pointCloud;
    bool configured=false;
};

constexpr int objectsPerCamera = 2;
using objectArray = std::array<object_t, objectsPerCamera>;

struct cameraConfig
{
    cv::Point3f camPose;
    cv::Mat img_base;
    objectArray objects;
    int camSpan;
};
