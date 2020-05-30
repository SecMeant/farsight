#pragma once

#include <cstdint>
#include <mutex>
#include <vector>

namespace farsight {
  struct Point3f
  {
    float x, y, z;
  };

  struct Context3D
  {
  public:
    void
    update(std::vector<Point3f> &&points, size_t width)
    {
      std::unique_lock lck{ this->mtx };

      this->points = std::move(points);
      this->width = width;
    }

    std::tuple<std::unique_lock<std::mutex>,
               const std::vector<Point3f> &,
               size_t>
    get_points() const
    {
      return { std::unique_lock(this->mtx), this->points, this->width };
    }

  private:
    mutable std::mutex mtx;
    size_t width = 1;
    std::vector<Point3f> points {{0,0,0}};
  };

} // namespace farsight
