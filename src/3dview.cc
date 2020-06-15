#include <cassert>
#include <cmath>
#include <mutex>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <fmt/format.h>

#include <GL/gl.h>
#include <GL/glut.h>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/rotate_vector.hpp>

#include "camera.h"
#include "types.h"
#include "utils.h"

using glm::cross;
using glm::dot;
using glm::normalize;
using glm::value_ptr;
using glm::vec3;

namespace farsight {
  static vec3 viewer = { 5.0, 0.0, 0.0 };
  static vec3 viewer_up = { 0.0, 1.0, 0.0 };
  static GLfloat mouse_sens = 1.0f / std::pow(2.0f, 9.0f);
  constexpr static GLfloat mouse_horizontal_max = M_PI * 2.0;
  constexpr static GLfloat mouse_horizontal_min = 0;
  constexpr static GLfloat mouse_vertical_max = M_PI / 2.0 - 0.1;
  constexpr static GLfloat mouse_vertical_min = -(M_PI / 2.0 - 0.1);
  static GLfloat mouse_horizontal = mouse_horizontal_max / 2.0;
  static GLfloat mouse_vertical =
    (mouse_vertical_max - mouse_vertical_min) / 2.0;
  static vec3 lookat = { 0.0, 0.0, 1.0f };
  static GLsizei XLOCK_POS = 256, YLOCK_POS = 256;
  static bool lock_mouse = true;

  static GLint status = 0;

  static int x_pos_old = 0;
  static int delta_x = 0;

  static int y_pos_old = 0;
  static int delta_y = 0;

  static GLfloat scale = 1;

  static GLfloat offset_x = 0;
  static GLfloat offset_y = 0;
  Context3D context3D;

  static void
  Axes(void)
  {
    vec3 x_min = { -50.0, 0.0, 0.0 };
    vec3 x_max = { 50.0, 0.0, 0.0 };

    vec3 y_min = { 0.0, -50.0, 0.0 };
    vec3 y_max = { 0.0, 50.0, 0.0 };

    vec3 z_min = { 0.0, 0.0, -50.0 };
    vec3 z_max = { 0.0, 0.0, 50.0 };

    glColor3f(1.0f, 0.0f, 0.0f);
    glBegin(GL_LINES);

    glVertex3fv(value_ptr(x_min));
    glVertex3fv(value_ptr(x_max));

    glEnd();

    glColor3f(0.0f, 0.5f, 0.0f);
    glBegin(GL_LINES);
    glVertex3fv(value_ptr(y_min));
    glVertex3f(0, 0, 0);
    glEnd();

    glColor3f(0.0f, 1.0f, 0.0f);
    glBegin(GL_LINES);
    glVertex3fv(value_ptr(y_max));
    glVertex3f(0, 0, 0);
    glEnd();

    glColor3f(1.0f, 1.0f, 1.0f);
    glBegin(GL_LINES);

    glVertex3fv(value_ptr(z_min));
    glVertex3fv(value_ptr(z_max));

    glEnd();
  }

  static void
  drawpoints(const CameraShot &cs)
  {
    float max_x = std::numeric_limits<float>::min(),
          max_y = std::numeric_limits<float>::min(),
          max_z = std::numeric_limits<float>::min(),
          min_x = std::numeric_limits<float>::max(),
          min_y = std::numeric_limits<float>::max(),
          min_z = std::numeric_limits<float>::max();

    glBegin(GL_POINTS);

    for (auto &p_ : cs.points)
    {
      glm::vec3 p = p_;
      auto color = p_.color;

      if (unlikely(std::isnan(p.z)))
        continue;

      p.x += cs.tvec.x;
      p.y += cs.tvec.y;
      p.z += cs.tvec.z;

      p = glm::rotateX(p, cs.rvec.x);
      p = glm::rotateY(p, cs.rvec.y);
      p = glm::rotateZ(p, cs.rvec.z);

      if (p.y <= (FLOOR_BASE_Y + cs.floor_level))
        continue;

      glColor3ub(color.r, color.g, color.b);
      glVertex3f(p.x, p.y, p.z);

      max_x = std::max(max_x, p.x);
      max_y = std::max(max_y, p.y);
      max_z = std::max(max_z, p.z);

      min_x = std::min(min_x, p.x);
      min_y = std::min(min_y, p.y);
      min_z = std::min(min_z, p.z);
    }

    fmt::print("Drawing points: \n"
               "\t max_x {}\n"
               "\t max_y {}\n"
               "\t max_z {}\n"
               "\t min_x {}\n"
               "\t min_y {}\n"
               "\t min_z {}\n",
               max_x,
               max_y,
               max_z,
               min_x,
               min_y,
               min_z);

    glEnd();
  }

  static void
  drawmarks(const RectArray &marks)
  {
    using VertType = decltype(marks[0].verts);

    for (auto &m : marks)
    {
      glBegin(GL_LINE_LOOP);

      VertType verts;
      std::copy(std::begin(m.verts), std::end(m.verts), std::begin(verts));

      auto color = m.color;

      for (auto &v : verts) {
        glColor3ub(color.r, color.g, color.b);
        glVertex3f(v.x, v.y, v.z);
      }

      glEnd();
    }
  }

  void
  RenderScene()
  {

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    gluLookAt(viewer[0],
              viewer[1],
              viewer[2],
              viewer[0] + lookat[0],
              viewer[1] + lookat[1],
              viewer[2] + lookat[2],
              viewer_up[0],
              viewer_up[1],
              viewer_up[2]);
    Axes();

    {
      auto [lck, cs] = context3D.get_points_cam1();
      drawpoints(cs);
    }

    {
      auto [lck, cs] = context3D.get_points_cam2();
      drawpoints(cs);
    }

    {
      auto [lck, marks] = context3D.get_marks();
      drawmarks(marks);
    }

    glFlush();
    glutSwapBuffers();
  }

  static void
  ChangeSize(GLsizei horizontal, GLsizei vertical)
  {
    XLOCK_POS = horizontal / 2;
    YLOCK_POS = vertical / 2;

    auto pix2anglex = 360.0 / (float)horizontal;
    auto pix2angley = 360.0 / (float)vertical;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(70, 1.0, 1.0, 300.0);

    if (horizontal <= vertical)
      glViewport(0, (vertical - horizontal) / 2, horizontal, horizontal);

    else
      glViewport((horizontal - vertical) / 2, 0, vertical, vertical);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
  }

  static void
  update_viewer_up()
  {
    viewer_up[0] = 0;
    viewer_up[1] = std::sin(mouse_vertical + (M_PI / 2.0));
    viewer_up[2] = 0;
  }

  static void
  update_lookat()
  {
    lookat[0] = std::cos(mouse_horizontal);
    lookat[1] = std::sin(mouse_vertical);
    lookat[2] = std::sin(mouse_horizontal);
  }

  static void
  Motion(GLsizei x, GLsizei y)
  {
    if (x == XLOCK_POS && y == YLOCK_POS)
      return;

    delta_x = x - XLOCK_POS;
    delta_y = YLOCK_POS - y;

    if (lock_mouse)
      glutWarpPointer(XLOCK_POS, YLOCK_POS);

    mouse_horizontal += delta_x * mouse_sens;
    mouse_vertical += delta_y * mouse_sens;

    if (mouse_horizontal > mouse_horizontal_max)
      mouse_horizontal = mouse_horizontal_min;
    else if (mouse_horizontal < mouse_horizontal_min)
      mouse_horizontal = mouse_horizontal_max;

    if (mouse_vertical > mouse_vertical_max)
      mouse_vertical = mouse_vertical_max;
    else if (mouse_vertical < mouse_vertical_min)
      mouse_vertical = mouse_vertical_min;

    update_lookat();
    update_viewer_up();

    glutPostRedisplay();
  }

  static void
  Keyboard(unsigned char key, int x, int y)
  {
    constexpr double camera_speed = 0.2f;
    static float tspeed = 0.05f;
    static float rotangle = (M_PI * 2.0f) / 32.0f;

    switch (key)
    {
      case 'm':
        mouse_sens /= 2.0f;
        break;

      case 'M':
        mouse_sens *= 2.0f;
        break;

      case '+':
        tspeed *= 2;
        rotangle *= 2;
        break;

      case '-':
        tspeed /= 2;
        rotangle /= 2;
        break;

      case 'q':
        lock_mouse = !lock_mouse;
        break;

      case 'w':
        viewer[0] += lookat[0] * camera_speed;
        viewer[1] += lookat[1] * camera_speed;
        viewer[2] += lookat[2] * camera_speed;
        update_lookat();
        update_viewer_up();
        break;

      case 's':
        viewer[0] -= lookat[0] * camera_speed;
        viewer[1] -= lookat[1] * camera_speed;
        viewer[2] -= lookat[2] * camera_speed;
        update_lookat();
        update_viewer_up();
        break;

      case 'd': {
        vec3 crossv;
        crossv = glm::cross(lookat, viewer_up);
        glm::normalize(crossv);
        viewer[0] += crossv[0] * camera_speed * 4.0;
        viewer[1] += crossv[1] * camera_speed * 4.0;
        viewer[2] += crossv[2] * camera_speed * 4.0;
        update_lookat();
        update_viewer_up();
        break;
      }

      case 'a': {
        vec3 crossv = cross(lookat, viewer_up);
        normalize(crossv);
        viewer[0] -= crossv[0] * camera_speed * 4.0;
        viewer[1] -= crossv[1] * camera_speed * 4.0;
        viewer[2] -= crossv[2] * camera_speed * 4.0;
        update_lookat();
        update_viewer_up();
        break;
      }

      case 'h':
      case 'j':
      case 'k':
      case 'l':
      case 'f':
      case 'g': {
        auto [lck, cs] = context3D.get_points_cam1();

        switch (key)
        {
          case 'h':
            cs.tvec.x -= tspeed;
            break;
          case 'l':
            cs.tvec.x += tspeed;
            break;

          case 'j':
            cs.tvec.y += tspeed;
            break;
          case 'k':
            cs.tvec.y -= tspeed;
            break;

          case 'f':
            cs.tvec.z += tspeed;
            break;
          case 'g':
            cs.tvec.z -= tspeed;
            break;
        }

        fmt::print(
          "CAM1 TVEC: {} {} {}\n", cs.tvec.x, cs.tvec.y, cs.tvec.z);
        break;
      }

      case 'H':
      case 'J':
      case 'K':
      case 'L':
      case 'F':
      case 'G': {
        auto [lck, cs] = context3D.get_points_cam2();

        switch (key)
        {
          case 'H':
            cs.tvec.x -= tspeed;
            break;
          case 'L':
            cs.tvec.x += tspeed;
            break;

          case 'J':
            cs.tvec.y += tspeed;
            break;
          case 'K':
            cs.tvec.y -= tspeed;
            break;

          case 'F':
            cs.tvec.z += tspeed;
            break;
          case 'G':
            cs.tvec.z -= tspeed;
            break;
        }

        fmt::print(
          "CAM2 TVEC: {} {} {}\n", cs.tvec.x, cs.tvec.y, cs.tvec.z);
        break;
      }

      case 'x':
      case 'y':
      case 'z': {
        auto [lck, cs] = context3D.get_points_cam1();

        switch (key)
        {
          case 'x':
            cs.rvec.x += rotangle;
            break;
          case 'y':
            cs.rvec.y += rotangle;
            break;
          case 'z':
            cs.rvec.z += rotangle;
            break;
        }

        fmt::print(
          "CAM1 ROT: {} {} {}\n", cs.rvec.x, cs.rvec.y, cs.rvec.z);
        break;
      }

      case 'X':
      case 'Y':
      case 'Z': {
        auto [lck, cs] = context3D.get_points_cam2();

        switch (key)
        {
          case 'X':
            cs.rvec.x += rotangle;
            break;
          case 'Y':
            cs.rvec.y += rotangle;
            break;
          case 'Z':
            cs.rvec.z += rotangle;
            break;
        }

        fmt::print(
          "CAM1 ROT: {} {} {}\n", cs.rvec.x, cs.rvec.y, cs.rvec.z);
        break;
      }

      default:
        break;
    }

    RenderScene();
  }

  void
  init3d()
  {
    int argc = 0;
    char *argv_ = nullptr;
    char **argv = &argv_;

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(300, 300);
    glutCreateWindow("3dview");

    glutDisplayFunc(RenderScene);
    glutReshapeFunc(ChangeSize);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glutPassiveMotionFunc(Motion);
    glutKeyboardFunc(Keyboard);
    glutSetCursor(GLUT_CURSOR_NONE);
    glutMainLoop();
  }
} // namespace farsight
