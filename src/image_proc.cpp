#include "image_proc.hpp"


void
rgbProcess(libfreenect2::Frame *frame)
{}

void
depthProcess(libfreenect2::Frame *frame)
{
  auto total_size = frame->height * frame->width;
  auto fp = reinterpret_cast<float *>(frame->data);

  for (int i = 0; i < total_size; i++)
  {
    fp[i] /= 4500.0f;
  }
}

cv::Mat frameDepthToMat(libfreenect2::Frame *frame)
{

}

cv::Mat frameRgbToMat(libfreenect2::Frame *frame)
{

}

