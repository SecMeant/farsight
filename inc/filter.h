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
    {}

    void
    apply(const libfreenect2::Frame &frame)
    {
      assert(frame.width == image.width);
      assert(frame.height == image.height);

      auto new_data = frame.data;
      auto data = image.data;

      for (auto i = frame.width * frame.height - 1; i >= 0; --i)
      {
        if (std::isnan(data[i]))
          data[i] = new_data[i];
      }
    }

    const FrameType&
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
}; // namespace postprocessing
