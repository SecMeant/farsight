#include <GL/gl.h>
#include <GL/glut.h>
#include <cmath>
#include <fmt/format.h>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

typedef float point3[3];

static GLfloat viewer[] = { 0.0, 0.0, 0.0 };
constexpr static GLfloat viewer_up[] = { 0.0, -1.0, 0.0 };
static GLfloat mouse_sens = 1.0f / std::pow(2.0f, 9.0f);
constexpr static GLfloat mouse_horizontal_max = M_PI * 2.0;
constexpr static GLfloat mouse_horizontal_min = 0;
constexpr static GLfloat mouse_vertical_max = M_PI * 2.0;
constexpr static GLfloat mouse_vertical_min = 0;
static GLfloat mouse_horizontal = mouse_horizontal_max / 2.0;
static GLfloat mouse_vertical = mouse_vertical_max / 2.0;
static GLfloat lookat[] = { 0.0, 0.0, 1.0f };
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

void
Axes(void)
{

  point3 x_min = { -5.0, 0.0, 0.0 };
  point3 x_max = { 5.0, 0.0, 0.0 };

  point3 y_min = { 0.0, -5.0, 0.0 };
  point3 y_max = { 0.0, 5.0, 0.0 };

  point3 z_min = { 0.0, 0.0, -5.0 };
  point3 z_max = { 0.0, 0.0, 5.0 };

  glColor3f(1.0f, 0.0f, 0.0f);
  glBegin(GL_LINES);

  glVertex3fv(x_min);
  glVertex3fv(x_max);

  glEnd();

  glColor3f(0.0f, 1.0f, 0.0f);
  glBegin(GL_LINES);

  glVertex3fv(y_min);
  glVertex3fv(y_max);

  glEnd();

  glColor3f(1.0f, 1.0f, 1.0f);
  glBegin(GL_LINES);

  glVertex3fv(z_min);
  glVertex3fv(z_max);

  glEnd();
}

float asdf = 128.0f;

const float *
matrix_at(const float *mat, size_t x, size_t y, size_t w)
{
  auto offset = x + y * w;
  auto pixel_ptr = static_cast<const float *>(mat + offset);
  return pixel_ptr;
}

void
drawpoints(const std::vector<float> &v)
{
  auto x = 0; // data_x;
  auto y = 0; // data_y;
  auto w = static_cast<size_t>(data_width);
  size_t i = 0;
  size_t j = 0;
  constexpr float scale_factor = 4.0f;
  const float *data = v.data();
  size_t h = v.size() / w;

  glBegin(GL_QUADS);
  glColor3f(0.75f, 0.0f, 0.35f);
  glVertex3f(0.0f, 0.0f, 0.0f);
  glVertex3f(h, 0.0f, 0.0f);
  glVertex3f(h, 0.0f, w);
  glVertex3f(0.0f, 0.0f, w);
  glEnd();

  glBegin(GL_POINTS);
  glColor3f(1.0f, 1.0f, 1.0f);

  auto cutoff = 0ul;

  for (size_t j = cutoff; j < h - 1 - cutoff; ++j)
    for (size_t i = cutoff; i < w - 1 - cutoff; ++i)
    {
      //fmt::print("Drawing at {} {} {}\n",
      //           *matrix_at(data, i, j, w),
      //           (y + j),
      //           (x + i));
      auto depth = *matrix_at(data, i, j, w);

      if (depth == 255.0f)
        continue;

      glVertex3f(depth / scale_factor,
                 (y + j) / scale_factor,
                 (x + i) / scale_factor);
    }
  // for (auto p : v)
  //{
  //  if (p == 255.0f || p == 0.0f)
  //    continue;
  //  glVertex3f(
  //    p / scale_factor, (y + j) / scale_factor, (x + i) / scale_factor);
  //  fmt::print("Drawing at {} {} {}\n",
  //             p / scale_factor,
  //             (y + j) / scale_factor,
  //             (x + i) / scale_factor);

  //  ++i;

  //  if (i == w)
  //  {
  //    i = 0;
  //    ++j;
  //  }
  //}

  glEnd();
}

void
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

  //glRotatef(thetax, 0.0, 1.0, 0.0);
  //glRotatef(thetay, 1.0, 0.0, 0.0);
  //glScalef(scale * 1.0, scale * 1.0, scale * 1.0);
  //glTranslatef(offset_x, offset_y, 0.0);

  Axes();

  drawpoints(points);
  // glutWireTeapot(3.0);

  glFlush();
  glutSwapBuffers();
}

void
MyInit(void)
{
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
}

void
ChangeSize(GLsizei horizontal, GLsizei vertical)
{

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

void
update_lookat()
{
  lookat[0] = std::cos(mouse_horizontal);
  lookat[1] = std::sin(mouse_vertical);
  lookat[2] = std::sin(mouse_horizontal);
}

void
Motion(GLsizei x, GLsizei y)
{
  if (x == 50 && y == 50)
    return;

  delta_x = x - x_pos_old;
  x_pos_old = x;

  delta_y = y - y_pos_old;
  y_pos_old = y;

  mouse_horizontal -= delta_x * mouse_sens;
  mouse_vertical -= delta_y * mouse_sens;

  if (mouse_horizontal > mouse_horizontal_max)
    mouse_horizontal = mouse_horizontal_max;
  else if (mouse_horizontal < mouse_horizontal_min)
    mouse_horizontal = mouse_horizontal_min;

  if (mouse_vertical > mouse_vertical_max)
    mouse_vertical = mouse_vertical_max;
  else if (mouse_vertical < mouse_vertical_min)
    mouse_vertical = mouse_vertical_min;

  update_lookat();

  glutPostRedisplay();

  fmt::print("Lookat: {} {} {}\n", lookat[0], lookat[1], lookat[2]);

  if (lock_mouse)
    glutWarpPointer(50,50);
}

constexpr void cross3fv(float *a, const float *b)
{
  float r[3] {};

  r[0] = a[1]*b[2]-a[2]*b[1];
  r[1] = a[2]*b[0]-a[0]*b[2];
  r[2] = a[0]*b[1]-a[1]*b[0];

  a[0] = r[0];
  a[1] = r[1];
  a[2] = r[2];
}

void cross3fv(float vR[], float v1[], float v2[]) {
  vR[0] =   ( (v1[1] * v2[2]) - (v1[2] * v2[1]) );
  vR[1] = - ( (v1[0] * v2[2]) - (v1[2] * v2[0]) );
  vR[2] =   ( (v1[0] * v2[1]) - (v1[1] * v2[0]) );
}

void norm3fv(float vR[], float v1[]) {
  float mag;

  mag = sqrt( pow(v1[0], 2) +
               pow(v1[1], 2) +
               pow(v1[2], 2)
             );

  vR[0] = v1[0] / mag;
  vR[1] = v1[1] / mag;
  vR[2] = v1[2] / mag;
}

void
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
      break;

    case 's':
      viewer[0] -= lookat[0] * camera_speed;
      viewer[1] -= lookat[1] * camera_speed;
      viewer[2] -= lookat[2] * camera_speed;
      update_lookat();
      break;

    case 'd': {
      float crossv[3];
      float crossvup[3] {viewer[0] + viewer_up[0], viewer[1] + viewer_up[1], viewer[2] + viewer_up[2]};
      cross3fv(crossv, viewer, crossvup);
      norm3fv(crossv, crossv);
      viewer[0] += crossv[0] * camera_speed;
      viewer[1] += crossv[1] * camera_speed;
      viewer[2] += crossv[2] * camera_speed;
      update_lookat();
      break;
    }

    case 'a': {
      float crossv[3];
      float crossvup[3] {viewer[0] + viewer_up[0], viewer[1] + viewer_up[1], viewer[2] + viewer_up[2]};
      cross3fv(crossv, viewer, crossvup);
      norm3fv(crossv, crossv);
      viewer[0] -= crossv[0] * camera_speed;
      viewer[1] -= crossv[1] * camera_speed;
      viewer[2] -= crossv[2] * camera_speed;
      update_lookat();
      break;
    }

    default:
      break;
  }

  fmt::print(
    "Position: {} {} {}\n"
    "Lookat: {} {} {}\n",
    viewer[0], viewer[1], viewer[2],
    lookat[0], lookat[1], lookat[2]);
  RenderScene();
}

void
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
  glutCreateWindow("Rzutowanie perspektywiczne");

  glutDisplayFunc(RenderScene);
  glutReshapeFunc(ChangeSize);
  MyInit();
  glEnable(GL_DEPTH_TEST);
  glutPassiveMotionFunc(Motion);
  glutKeyboardFunc(Keyboard);
  glutMainLoop();
}
