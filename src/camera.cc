#include <cassert>

#include "camera.h"

namespace farsight {

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc99-designator"

  // clang-format off
  constexpr glm::vec3 faces_position[6] {
    [0] = { 0.00f, 0.00f, 0.00f },
    [1] = { 0.00f, 0.25f,-0.25f },
    [2] = { 0.00f, 0.00f,-0.50f },
    [3] = {-0.25f, 0.00f,-0.25f },
    [4] = { 0.25f, 0.00f,-0.25f },
    [5] = { 0.00f,-0.25f,-0.25f },

  };
  // clang-format on

#pragma clang diagnostic pop
#endif

  constexpr glm::vec3 front_face = faces_position[1];

  void
  camera2real(std::vector<Point3f> &points,
              glm::vec3 tvec,
              glm::mat3x3 rot,
              int id)
  {
    assert(id < std::size(faces_position) && id >= 0);

    tvec = tvec * (-1.0f);

    for (auto &point : points) {
      point.x += tvec.x;
      point.y += tvec.y;
      point.z += tvec.z;

      point = rot * point;
    }
  }

} // namespace farsight
