#pragma once

#include <cstdint>
#include <mutex>
#include <vector>

#include <glm/glm.hpp>

namespace farsight {
  struct Point3f
  {
    float x, y, z;
  };

  struct CameraShot
  {
    size_t width = 1;
    std::vector<Point3f> points {{0,0,0}};
    glm::vec3 tvec {0.0f, 0.0f, 0.0f};
  };

  struct Context3D
  {
  public:
    using PointInfoLocked = std::tuple<std::unique_lock<std::mutex>, CameraShot&>;
    void
    update_cam1(std::vector<Point3f> &&points, size_t width)
    {
      std::unique_lock lck{ this->mtx };

      this->camshot1.points = std::move(points);
      this->camshot1.width = width;
    }

    void
    update_cam2(std::vector<Point3f> &&points, size_t width)
    {
      std::unique_lock lck{ this->mtx };

      this->camshot2.points = std::move(points);
      this->camshot2.width = width;
    }

    PointInfoLocked
    get_points_cam1()
    {
      return { std::unique_lock(this->mtx), this->camshot1 };
    }

    PointInfoLocked
    get_points_cam2()
    {
      return { std::unique_lock(this->mtx), this->camshot2 };
    }

  private:
    mutable std::mutex mtx;
    CameraShot camshot1, camshot2;
  };

} // namespace farsight
