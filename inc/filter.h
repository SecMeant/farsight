#pragma once

#include <cassert>
#include <cmath>

#include <libfreenect2/frame_listener.hpp>

namespace farsight::postprocessing {

  using PixelType = float;
  using FrameType = libfreenect2::Frame;

  struct Stage1
  {
    Stage1(size_t width, size_t height)
      : image(width, height, sizeof(PixelType))
    {
      reset();
    }

    void
    apply(const libfreenect2::Frame &frame)
    {
      assert(frame.width == image.width);
      assert(frame.height == image.height);

      auto new_data = reinterpret_cast<float*>(frame.data);
      auto data = reinterpret_cast<float*>(image.data);

      for (size_t i = 0; i < frame.width * frame.height ; ++i)
      {
        if (std::isnan(data[i]) || data[i] < 0.0f || std::isinf(data[i]))
          data[i] = new_data[i];
      }
    }
    void
    reset()
    {
      auto *data = reinterpret_cast<PixelType *>(image.data);
      for (int i = 0; i < image.width * image.height; i++)
      {
        data[i] = NAN;
      }
    }

    const FrameType &
    get()
    {
      return image;
    }

    FrameType image;
  };

  void
  blur_at(FrameType &frame, size_t x, size_t y);

  void
  blur(FrameType &frame);
}; // namespace farsight::postprocessing
