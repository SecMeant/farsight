#pragma once
#include <climits>
#include <libfreenect2/libfreenect2.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

using byte = unsigned char;

struct position
{
    double x, y, z;
};

struct bbox
{
  int x = 0, y = 0, w = 0, h = 0, area = 0;
  void
  operator+=(const bbox &b)
  {
    w += b.w;
    h += b.h;
    area += b.area;
  }

  void
  reset()
  {
    x = y = w = h = area = 0;
  }
};

void
conv8UC4To32FC4(byte *, size_t);
void
conv8UC4To32FC1(byte *, size_t);
void
conv32FC1To8CU1(byte *, size_t);

void
gamma(float *, size_t, float);
void
removeAlpha(float *, size_t);
void
diff(byte *, byte *, size_t, char th = 10);
void
rgbProcess(libfreenect2::Frame *frame);
void
depthProcess(libfreenect2::Frame *frame);

// byte, short, int, float ...
template<typename Format>
position
findNearestPoint(const bbox &area, const byte *frame)
{
  constexpr int mm_to_cm = 10;
  position dep;
  int pos = area.y * area.w + area.x;
  const auto *depth = reinterpret_cast<const Format *>(frame);

  for (int i = 0; i < area.w * area.h; i++)
  {
    auto x = (area.x + (i % area.w));
    auto y = (area.y + (i / area.w));
    pos = y * area.w + x;
    if (dep.z > depth[pos] and depth[pos] > 0.0 and
        !std::isnan(depth[pos]))
    {
      dep.z= fabs(depth[pos]);
      dep.x= x;
      dep.y= y * area.w;
    }
  }
  dep.z= dep.z/ mm_to_cm; // convert to cm
  return dep;
}
