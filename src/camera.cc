#include "camera.h"

namespace farsight {

  // clang-format off
  glm::vec3 faces_position[6] {
    [0] = { 0.00f, 0.00f, 0.00f },
    [1] = { 0.00f, 0.25f,-0.25f },
    [2] = { 0.00f, 0.00f,-0.50f },
    [3] = {-0.25f, 0.00f,-0.25f },
    [4] = { 0.25f, 0.00f,-0.25f },
    [5] = { 0.00f,-0.25f,-0.25f },

  };
  // clang-format on

  void
  camera2real(std::vector<Point3f> &points,
              glm::vec3 tvec,
              glm::mat3x3 rot,
              int id)
  {
    tvec = tvec * (-1.0f);

    for (auto &point : points) {
      point.x += tvec.x;
      point.y += tvec.y;
      point.z += tvec.z;

     // auto rotated = rot * glm::vec3{point.x, point.y, point.z};

     // point.x = rotated.x;
     // point.y = rotated.y;
     // point.z = rotated.z;
    }
  }

} // namespace farsight
