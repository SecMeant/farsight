#pragma once

#include <vector>

#include <glm/glm.hpp>

#include "types.h"

namespace farsight {
  void
  camera2real(PointArray &points,
              glm::vec3 tvec,
              glm::mat3x3 rvec = glm::mat3x3(1),
              int id = 0
              );
}
