#include "image_utils.hpp"

void conv8UC4To32FC1(byte *data, size_t size)
{
  auto rgbend = data + size;
  auto rgbp = rgbend - (size / sizeof(float));

  for (auto rgbfp = reinterpret_cast<float *>(data); rgbp != rgbend;
       rgbp+=4, ++rgbfp)
  {
    auto b = static_cast<float>(rgbp[0]) / 255.0f;
    auto g = static_cast<float>(rgbp[1]) / 255.0f;
    auto r = static_cast<float>(rgbp[2]) / 255.0f;

    *rgbfp = (b + g + r ) / 3.0f;
  }
}

void conv32FC1To8CU1(byte *data, size_t size)
{
  auto fp = reinterpret_cast<float*>(data);

  for (auto i = 0; i < size; ++i, ++fp, ++data)
    *data = static_cast<byte>(*fp * 255.0f);
}

void
conv8UC4To32FC4(byte *data, size_t size)
{
  auto rgbend = data + size;
  auto rgbp = rgbend - (size / sizeof(float));

  for (auto rgbfp = reinterpret_cast<float *>(data); rgbp != rgbend;
       ++rgbp, ++rgbfp)
  {
    *rgbfp = static_cast<float>(*rgbp) / 255.0f;
  }
}

void
removeAlpha(float *data, size_t size)
{
  assert(size % 4 == 0);

  for (auto current = 0, insert = 0; current < size;
       current += 4, insert += 3)
  {
    data[insert + 0] = data[current + 0];
    data[insert + 1] = data[current + 1];
    data[insert + 2] = data[current + 2];
  }
}

void
gamma(float *data, size_t size, float gamma)
{
  for (auto i = 0; i < size; ++i)
    data[i] = std::pow(data[i], 1.0f / gamma);
}

void diff(byte *i1_, byte *i2_, size_t size, char th)
{
  auto i1 = reinterpret_cast<char*>(i1_);
  auto i2 = reinterpret_cast<char*>(i2_);

  for (auto i = 0; i < size; ++i)
  {
    if (std::abs(i1[i] - i2[i]) < th)
      i1[i] = 255;
  }
}

