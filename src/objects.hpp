#pragma once
#include "image_utils.hpp"
#include "config.hpp"
#include <type_traits>
using pointArray = std::vector<cv::Point2f>;

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
    bbox realArea;
    position nearest_point;
    cv::Mat imgDepth = cv::Mat::zeros(
        cv::Size(depth_width, depth_height), CV_8UC1); 
    std::unique_ptr<byte[]> depthFrame =
      std::make_unique<byte[]>(total_size_depth);
    pointArray flattenedObject;
    bool configured=false;
};

constexpr int objectsPerCamera = 2;
using objectArray = std::array<object_t, objectsPerCamera>;

struct cameraConfig
{
    position camPose;
    cv::Mat img_base;
    objectArray objects;
    int camSpan;
};
