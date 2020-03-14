#pragma once
#include <libfreenect2/libfreenect2.hpp>
#include <opencv2/core/mat.hpp>
#include "image_utils.hpp"

void rgbProcess(libfreenect2::Frame *frame);
void depthProcess(libfreenect2::Frame *frame);
void detectObject(cv::Mat&);

