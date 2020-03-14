#pragma once
#include <libfreenect2/libfreenect2.hpp>
#include <opencv2/core/mat.hpp>
#include "image_utils.hpp"

void rgbProcess(libfreenect2::Frame *frame);
void depthProcess(libfreenect2::Frame *frame);
cv::Mat frameDepthToMat(libfreenect2::Frame *frame);
cv::Mat frameRgbToMat(libfreenect2::Frame *frame);

void detectObject();

