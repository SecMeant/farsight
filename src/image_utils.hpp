#pragma ocne
#include <climits>
#include <libfreenect2/libfreenect2.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

using byte = unsigned char;
struct depth_t
{
  static inline constexpr double maxDepth =
    std::numeric_limits<double>::max();
  int pixel_x = 0, pixel_y = 0;
  double depth = maxDepth;
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
depth_t
scopeMin(const bbox &area, const byte *frame)
{
  constexpr int ms_to_s = 1000;
  depth_t dep;
  int pos = area.y * area.w + area.x;
  const auto *depth = reinterpret_cast<const Format *>(frame);

  for (int i = 0; i < area.w * area.h; i++)
  {
    auto x = (area.x + (i % area.w));
    auto y = (area.y + (i / area.w));
    pos = y * area.w + x;
    if (dep.depth > depth[pos] and depth[pos] > 0.0 and
        !std::isnan(depth[pos]))
    {
      dep.depth = fabs(depth[pos]);
      dep.pixel_x = x;
      dep.pixel_y = y * area.w;
    }
  }
  dep.depth = dep.depth / ms_to_s; // convert to cm
  return dep;
}
