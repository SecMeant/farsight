#include <cmath>
#include <cassert>

#include "camera.h"

#include <fmt/format.h>

constexpr static float M_TAU = M_PI * 2.0f;

namespace farsight {

  enum class Axis {
    X, Y, Z
  };

  static glm::mat3x3 rotmat(Axis axis, float angle)
  {
    // clang-format off
    switch(axis) {
      case Axis::X:
        return glm::mat3x3(
           1.0f            ,  0.0f            , 0.0f           ,
           0.0f            ,  std::cos(angle) , std::sin(angle),
           0.0f            , -std::sin(angle) , std::cos(angle)
        );

      case Axis::Y:
        return glm::mat3x3(
           std::cos(angle) , 0.0f             , -std::sin(angle),
           0.0f            , 1.0f             , 0.0f            ,
           std::sin(angle) , 0.0f             , std::cos(angle)
        );

      case Axis::Z:
        return glm::mat3x3(
           std::cos(angle)  , std::sin(angle) , 0.0f            ,
          -std::sin(angle)  , std::cos(angle) , 0.0f            ,
           0.0f             , 0.0f            , 1.0f
        );
    }
    // clang-format on

    assert(false && "Unexpected enum value");
    return glm::mat3x3(1.0f);
  }

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc99-designator"
#endif

  // clang-format off
  const glm::vec3 faces_position[6] {
    [0] = glm::vec3( 0.25f, 0.00f, 0.25f ),
    [1] = glm::vec3(-0.00f, 0.00f, 0.00f ),
    [2] = glm::vec3(-0.25f, 0.00f, 0.25f ),
    [3] = glm::vec3(-0.00f,-0.25f, 0.25f ),
    [4] = glm::vec3(-0.00f, 0.25f, 0.25f ),
    [5] = glm::vec3(-0.00f, 0.00f, 0.50f ),
  };

  const glm::mat3x3 faces_rotation[6] {
    [0] = rotmat(Axis::Y, M_TAU / 4.0f),
    [1] = glm::mat4x4(1.0f),
    [2] = rotmat(Axis::Y, -(M_TAU / 4.0f)),
    [3] = rotmat(Axis::X, M_TAU / 4.0f),
    [4] = rotmat(Axis::X, -(M_TAU / 4.0f)),
    [5] = rotmat(Axis::Y, M_TAU / 2.0f)
  };

  // clang-format on

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

  const glm::vec3 front_face = faces_position[1];

  constexpr static void
  check_face_id(int id)
  {
    // 3 is bottom face and cannot be seen
    assert(id < std::size(faces_position) && id >= 0 && id != 3);
  }

  static glm::vec3
  calculate_face_offset(int id)
  {
    check_face_id(id);
    return faces_position[id] * -1.0f;
  }

  static glm::mat3x3
  calculate_face_rotation(int id)
  {
    check_face_id(id);
    return faces_rotation[id];
  }

  void
  camera2real(PointArray &points,
              glm::vec3 tvec,
              glm::mat3x3 rot,
              int id)
  {
    check_face_id(id);

    // Get found marker offset from camera
    glm::vec3 camera_pos = tvec;

    // Apply relative face offset from camera to front marker
    camera_pos += calculate_face_offset(id);

    // Get camera position relative to front marker (world 0,0,0)
    camera_pos *= -1.0f;

    fmt::print("Camera pos: {} {} {} {}\n", camera_pos.x, camera_pos.y, camera_pos.z, id);

    // Fix badly printed aruco?
    rot = rot * rotmat(Axis::Z, -(M_TAU / 4.0f));

    // Apply camera offset and rotation to all points.
    for (auto &point : points) {
      point += camera_pos;

      // Apply found maker rotation
      point = rot * point;

      // Apply marker relative rotation
      point = calculate_face_rotation(id) * point;
    }
  }

} // namespace farsight
