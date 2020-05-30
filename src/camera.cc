#include "camera.h"

namespace farsight {
  void
  camera2real(std::vector<Point3f> &points,
              glm::vec3 tvec,
              glm::mat3x3 rot)
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
