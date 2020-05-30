#include <cassert>
#include <cmath>
#include <vector>
#include <mutex>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <fmt/format.h>

#include <GL/gl.h>
#include <GL/glut.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "camera.h"
#include "types.h"
#include "utils.h"

using glm::vec3;
using glm::value_ptr;
using glm::normalize;
using glm::cross;
using glm::dot;

namespace farsight
{
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

static const float *
matrix_at(const float *mat, size_t x, size_t y, size_t w)
{
  auto offset = x + y * w;
  auto pixel_ptr = static_cast<const float *>(mat + offset);
  return pixel_ptr;
}

static Point3f
matrix_point_at(const Point3f *mat, size_t x, size_t y, size_t w)
{
  auto offset = x + y * w;
  return *(reinterpret_cast<const Point3f *>(mat) + offset);
}

static void
drawpoints(const CameraShot &cs)
{
  auto w = cs.width;
  size_t i = 0;
  size_t j = 0;
  constexpr float scale_factor_depth = 1.0f;
  constexpr float scale_factor_width = 1.0f;
  constexpr float scale_factor_height = 1.0f;
  size_t h = cs.points.size() / w;

  glBegin(GL_POINTS);
  glColor3f(1.0f, 1.0f, 1.0f);

  for (size_t j = 0; j < h - 1; ++j)
    for (size_t i = 0; i < w - 1; ++i)
    {
      auto p = matrix_point_at(cs.points.data(), i, j, w);

      if (std::isnan(p.z))
        continue;

      float x = p.x * scale_factor_width, y = p.y * scale_factor_height,
            z = p.z * scale_factor_depth;

      x += cs.tvec.x;
      y += cs.tvec.y;
      z += cs.tvec.z;

      y *= (-1.0f);

      glVertex3f(x,y,z);
    }

  glEnd();
}

static void
RenderScene(void)
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

  auto [_, points, width] = context3D.get_points_cam1();
  drawpoints(points, width);

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
  float tspeed = 0.05f;

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
      break;

    case '-':
      tspeed /= 2;
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
    case 'y':
    case 'n': {
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

        case 'y':
          cs.tvec.z += tspeed;
          break;
        case 'n':
          cs.tvec.z -= tspeed;
          break;
      }

      fmt::print("CAM1 TVEC: {} {} {}\n", cs.tvec.x, cs.tvec.y, cs.tvec.z);
    }

    case 'H':
    case 'J':
    case 'K':
    case 'L':
    case 'Y':
    case 'N': {
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

        case 'Y':
          cs.tvec.z += tspeed;
          break;
        case 'N':
          cs.tvec.z -= tspeed;
          break;
      }

      fmt::print("CAM2 TVEC: {} {} {}\n", cs.tvec.x, cs.tvec.y, cs.tvec.z);
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
} // nampespace farsight
