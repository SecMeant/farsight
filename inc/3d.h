#pragma once

#include <cstdint>
#include <vector>

#include "types.h"

extern "C" {
void glutPostRedisplay();
}

namespace farsight {

  extern Context3D context3D;

  void
  init3d();

  inline void
  update_points_cam1(std::vector<farsight::Point3f> points, size_t width)
  {
    context3D.update_cam1(std::move(points), width);
  }

  inline void
  update_points_cam2(std::vector<farsight::Point3f> points, size_t width)
  {
    context3D.update_cam2(std::move(points), width);
  }

  inline glm::vec3
  get_tvec_cam1()
  {
    return context3D.get_tvec_cam1();
  }

  inline glm::vec3
  get_rvec_cam1()
  {
    return context3D.get_rvec_cam1();
  }

  inline glm::vec3
  get_tvec_cam2()
  {
    return context3D.get_tvec_cam2();
  }

  inline glm::vec3
  get_rvec_cam2()
  {
    return context3D.get_rvec_cam2();
  }

  inline void
  set_tvec_cam1(glm::vec3 v)
  {
    context3D.set_tvec_cam1(v);
  }

  inline void
  set_rvec_cam1(glm::vec3 v)
  {
    context3D.set_rvec_cam1(v);
  }

  inline void
  set_tvec_cam2(glm::vec3 v)
  {
    context3D.set_tvec_cam2(v);
  }

  inline void
  set_rvec_cam2(glm::vec3 v)
  {
    context3D.set_rvec_cam2(v);
  }

  inline Context3D::PointInfo
  get_translated_points_cam1()
  {
    return context3D.get_translated_points_cam1();
  }

  inline Context3D::PointInfo
  get_translated_points_cam2()
  {
    return context3D.get_translated_points_cam2();
  }

  inline void
  set_floor_level(float f)
  {}

  inline float
  get_floor_level()
  {return 0.0;}

} // namespace farsight
