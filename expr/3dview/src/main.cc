#include <cassert>
#include <cmath>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <fmt/format.h>

#include <GL/gl.h>
#include <GL/glut.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

using glm::vec3;
using glm::value_ptr;
using glm::normalize;
using glm::cross;
using glm::dot;

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

float data_x, data_y, data_width;
std::vector<float> points;

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

float asdf = 128.0f;

static const float *
matrix_at(const float *mat, size_t x, size_t y, size_t w)
{
  auto offset = x + y * w;
  auto pixel_ptr = static_cast<const float *>(mat + offset);
  return pixel_ptr;
}

static void
drawpoints(const std::vector<float> &v)
{
  auto x = 0; // data_x;
  auto y = 0; // data_y;
  auto w = static_cast<size_t>(data_width);
  size_t i = 0;
  size_t j = 0;
  constexpr float scale_factor_depth = 4.0f;
  constexpr float scale_factor_width = 12.0f;
  constexpr float scale_factor_height = 12.0f;
  const float *data = v.data();
  size_t h = v.size() / w;

  glBegin(GL_POINTS);
  glColor3f(1.0f, 1.0f, 1.0f);

  auto cutoff = 0ul;

  for (size_t j = 0; j < h - 1 - cutoff; ++j)
    for (size_t i = cutoff; i < w - 1 - cutoff; ++i)
    {
      auto depth = *matrix_at(data, i, h - 1 - cutoff - j, w);

      if (depth == 255.0f)
        continue;

      glVertex3f((x + i) / scale_factor_width,
                 (y + j) / scale_factor_height,
                 depth / scale_factor_depth);
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

  drawpoints(points);

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
  constexpr double camera_speed = 1.0f;

  switch (key)
  {
    case 'm':
      mouse_sens /= 2.0f;
      break;

    case 'M':
      mouse_sens *= 2.0f;
      break;

    case 'l':
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

    default:
      break;
  }

  RenderScene();
}

static void
usage(int argc, char **argv)
{
  fmt::print("Usage: {} <raw data file>\n", argv[0]);
}

int
read_all(int fd, void *output, size_t size)
{
  auto begin = static_cast<char *>(output);
  auto end = begin + size;

  while (begin != end)
  {
    auto ret = read(fd, begin, end - begin);

    if (ret <= 0)
    {
      perror("Error while reading file");
      return ret;
    }

    begin += ret;
  }

  return 0;
}

int
main(int argc, char **argv)
{

  if (argc != 2)
  {
    usage(argc, argv);
    return 1;
  }

  int fd = open(argv[1], 0, O_RDONLY);

  if (fd == -1)
  {
    perror(fmt::format("Failed to open {}", argv[1]).c_str());
    return 2;
  }

  struct stat fstats;
  if (fstat(fd, &fstats))
  {
    close(fd);
    perror("Failed to get info about opened file");
    return 3;
  }

  float data_header[3];
  if (read_all(fd, &data_header, sizeof(data_header)))
  {
    close(fd);
    perror("Error while reading file");
    return 4;
  }

  auto raw_data_size = fstats.st_size - sizeof(data_header);
  auto data_count = raw_data_size / sizeof(float);

  points.resize(data_count);
  char *output = reinterpret_cast<char *>(points.data());

  if (read_all(fd, output, raw_data_size))
  {
    // close(fd); // let kernel clean this up
    perror("Error while reading file");
    return 4;
  }

  close(fd);

  data_x = data_header[0];
  data_y = data_header[1];
  data_width = data_header[2];

  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  glutInitWindowSize(300, 300);
  glutCreateWindow("kinect 3dview");

  glutDisplayFunc(RenderScene);
  glutReshapeFunc(ChangeSize);
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  glEnable(GL_DEPTH_TEST);
  glutPassiveMotionFunc(Motion);
  glutKeyboardFunc(Keyboard);
  glutSetCursor(GLUT_CURSOR_NONE);
  glutMainLoop();
}
