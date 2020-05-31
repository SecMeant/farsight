#include <cassert>

#include "filter.h"

using Format = libfreenect2::Frame::Format;

#define blur_accumulate_if(frame, x, y)                                   \
  if (auto p = frame_at_checked<float>(frame, x, y); p != nullptr && *p != 0.0f) \
  {                                                                       \
    sum += *p;                                                            \
    ++count;                                                              \
  }

namespace farsight::postprocessing {
  template<typename T>
  T *
  frame_at(const FrameType &f, size_t x, size_t y)
  {
    assert(x >= 0 && x < f.width && y >= 0 && y < f.height);
    assert(f.format == Format::Float);
    float *data = reinterpret_cast<float *>(f.data);

    return data + x + y * f.width;
  }

  template<typename T>
  T *
  frame_at_checked(const FrameType &f, size_t x, size_t y)
  {
    assert(f.format == Format::Float);

    if (x < 0 || x >= f.width || y < 0 || y >= f.height)
      return nullptr;

    float *data = reinterpret_cast<float *>(f.data);
    return data + x + y * f.width;
  }

    void
    blur_at(FrameType &frame, size_t x, size_t y)
    {
      size_t sum = 0, count = 0;

      if (*frame_at<float>(frame, x, y) != 0.0f)
        return;

      blur_accumulate_if(frame, x - 1, y - 1);
      blur_accumulate_if(frame, x, y - 1);
      blur_accumulate_if(frame, x + 1, y - 1);

      blur_accumulate_if(frame, x - 1, y);
      blur_accumulate_if(frame, x + 1, y);

      blur_accumulate_if(frame, x - 1, y + 1);
      blur_accumulate_if(frame, x, y + 1);
      blur_accumulate_if(frame, x + 1, y + 1);

      auto blur = sum / count;
      assert(blur <= std::numeric_limits<uint8_t>::max());

      *frame_at<float>(frame, x, y) = blur;
    }

    void
    blur(FrameType &frame)
    {
      auto width = frame.width;
      auto height = frame.height;

      for (size_t j = 0; j < height; ++j)
        for (size_t i = 0; i < width; ++i)
          blur_at(frame, i, j);
    }

  }; // namespace farsight::postprocessing

#undef blur_accumulate_if
