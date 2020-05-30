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
    glutPostRedisplay();
  }

  inline void
  update_points_cam2(std::vector<farsight::Point3f> points, size_t width)
  {
    context3D.update_cam2(std::move(points), width);
    glutPostRedisplay();
  }

} // namespace farsight
