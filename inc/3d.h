#pragma once

#include <cstdint>
#include <vector>

#include "types.h"

namespace farsight {

  extern Context3D context3D;

  void
  init3d();

  void
  update_points(std::vector<farsight::Point3f> points, size_t width)
  {
    context3D.update(std::move(points), width);
  }

} // namespace farsight
