#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define unlikely(x) __builtin_expect(x, 0)
#define likely(x) __builtin_expect(x, 1)

const char *wndname = "wnd";
const char *wndname2 = "wnd2";
const char *wndname3 = "wnd3";
const char *wndname4 = "wnd4";

std::atomic_flag continue_flag;

libfreenect2::SyncMultiFrameListener listener(
  libfreenect2::Frame::Color | libfreenect2::Frame::Ir |
  libfreenect2::Frame::Depth);
libfreenect2::FrameMap frames;
libfreenect2::Freenect2Device *dev;
 libfreenect2::Freenect2 freenect2;
extern "C"
{
#include <signal.h>
#include <unistd.h>
}

using namespace cv;

struct bbox
{
  int x, y, w, h, area;
};

struct shared_t
{
  std::mutex lock;
  cv::Mat &depth_image;
  bbox &best_bbox;
};

int
write_all(int fd, void *data, size_t size)
{
  auto begin = static_cast<char *>(data);
  auto end = begin + size;

  while (unlikely(begin != end))
  {
    auto ret = write(fd, begin, end - begin);

    if (ret <= 0)
      return ret;

    begin += ret;
  }

  return 0;
}

static auto frame_outfile = "frame";

uint8_t *
matrix_at(const cv::Mat &mat, size_t x, size_t y)
{
  auto rows = mat.rows;
  auto cols = mat.cols;

  assert(x <= cols && x >= 0);
  assert(y <= rows && y >= 0);

  auto offset = x + y * cols;
  auto pixel_ptr = static_cast<uint8_t *>(mat.data + offset);

  return pixel_ptr;
}

uint8_t *
matrix_at_checked(const cv::Mat &mat, size_t x, size_t y)
{
  auto rows = mat.rows;
  auto cols = mat.cols;

  if (unlikely(x >= cols) || unlikely(y >= rows))
    return nullptr;

  auto offset = x + y * cols;
  auto pixel_ptr = static_cast<uint8_t *>(mat.data + offset);

  return pixel_ptr;
}

namespace postprocessing {

  using CvType = int;

  template<CvType cvtype_>
  struct NAN_PIXEL;

  template<>
  struct NAN_PIXEL<CV_8UC1>
  {
    constexpr static uint8_t value = 255;
  };

  template<CvType cvtype_>
  struct Stage1
  {
    constexpr static auto cvtype = cvtype_;
    constexpr static auto nan_pixel = NAN_PIXEL<cvtype_>::value;

    Stage1() = default;

    Stage1(size_t width, size_t height)
      : image(width, height, cvtype)
    {}

    void
    resize(size_t width, size_t height)
    {
      image.resize(width, height);
    }

    void
    apply(const cv::Mat &mat)
    {
      assert(mat.rows == image.rows);
      assert(mat.cols == image.cols);

      auto new_data = mat.data;
      auto data = image.data;

      for (auto i = mat.rows * mat.cols - 1; i >= 0; --i)
      {
        if (data[i] == nan_pixel)
          data[i] = new_data[i];
      }
    }

    cv::Mat
    get()
    {
      return image;
    }

    cv::Mat image;
  };

#define blur_accumulate_if(mat, x, y)                                     \
  if (auto p = matrix_at_checked(mat, x, y); p != nullptr && *p != 0.0f)  \
  {                                                                       \
    sum += *p;                                                            \
    ++count;                                                              \
  }

  void
  blur_at(cv::Mat &mat, size_t x, size_t y)
  {
    size_t sum = 0, count = 0;

    if (*matrix_at(mat, x, y) != 0.0f)
      return;

    blur_accumulate_if(mat, x - 1, y - 1);
    blur_accumulate_if(mat, x, y - 1);
    blur_accumulate_if(mat, x + 1, y - 1);

    blur_accumulate_if(mat, x - 1, y);
    blur_accumulate_if(mat, x + 1, y);

    blur_accumulate_if(mat, x - 1, y + 1);
    blur_accumulate_if(mat, x, y + 1);
    blur_accumulate_if(mat, x + 1, y + 1);

    auto blur = sum / count;
    assert(blur <= std::numeric_limits<uint8_t>::max());

    *matrix_at(mat, x, y) = blur;
  }

  void
  blur_(cv::Mat &mat)
  {
    auto width = mat.cols;
    auto height = mat.rows;

    for (size_t j = 0; j < height; ++j)
      for (size_t i = 0; i < width; ++i)
        blur_at(mat, i, j);
  }

  void
  blur(cv::Mat &mat, int ksize = 5)
  {
    cv::medianBlur(mat, mat, ksize);
  }

}; // namespace postprocessing

void
mouse_event_handler(int event, int x, int y, int flags, void *userdata)
{
  shared_t *shared = static_cast<shared_t *>(userdata);

  if (event == cv::EVENT_LBUTTONDOWN)
  {
    std::scoped_lock lck(shared->lock);
    auto pixel_ptr = matrix_at(shared->depth_image, x, y);
    fmt::print("Value: {}\n", *pixel_ptr);
  }
  else if (event == cv::EVENT_RBUTTONDOWN)
  {
    fmt::print("Dumping bbox image\n");
    std::scoped_lock lck(shared->lock);

    int fd = open(frame_outfile, O_CREAT | O_WRONLY, 0660);

    if (fd == -1)
    {
      perror("Failed to open file for dumping data");
      return;
    }

    float row_data_coords[] = { static_cast<float>(shared->best_bbox.x),
                                static_cast<float>(shared->best_bbox.y),
                                static_cast<float>(shared->best_bbox.w) };

    if (write_all(fd, row_data_coords, sizeof(row_data_coords)))
    {
      perror("Error while writing data to file");
      return;
    }

    float row_data_float[shared->best_bbox.w];

    for (auto col = 0; col < shared->best_bbox.h; ++col)
    {
      auto x = shared->best_bbox.x, y = shared->best_bbox.y;
      auto row_data = matrix_at(shared->depth_image, x, y + col);

      for (auto i = 0; i < shared->best_bbox.w; ++i)
        row_data_float[i] = static_cast<float>(row_data[i]);

      if (write_all(fd, row_data_float, sizeof(row_data_float)))
      {
        perror("Error while writing data to file");
        return;
      }
    }

    close(fd);
  }
  else if (event == cv::EVENT_MBUTTONDOWN)
  {
    fmt::print("Dumping whole image\n");
    std::scoped_lock lck(shared->lock);

    int fd = open(frame_outfile, O_CREAT | O_WRONLY, 0660);

    if (fd == -1)
    {
      perror("Failed to open file for dumping data");
      return;
    }

    auto width = shared->depth_image.cols;
    auto height = shared->depth_image.rows;

    float row_data_coords[] = {0, 0, static_cast<float>(width)};

    if (write_all(fd, row_data_coords, sizeof(row_data_coords)))
    {
      perror("Error while writing data to file");
      return;
    }

    float row_data_float[width];

    for (auto col = 0; col < height; ++col)
    {
      auto x = 0, y = 0;
      auto row_data = matrix_at(shared->depth_image, x, y + col);

      for (auto i = 0; i < width; ++i)
        row_data_float[i] = static_cast<float>(row_data[i]);

      if (write_all(fd, row_data_float, sizeof(row_data_float)))
      {
        perror("Error while writing data to file");
        return;
      }
    }

    close(fd);
  }
}

void
sigint_handler(int signo)
{
  fmt::print("Signal handler\n");

  if (signo == SIGINT)
  {
    fmt::print("Got SIGINT\n");
    continue_flag.clear();
  }
}
using byte = unsigned char;


constexpr size_t cubeface_count = 6;
std::array<cv::Mat, cubeface_count> cubefaces;

void
gamma(float *data, size_t size, float gamma)
{
  for (auto i = 0; i < size; ++i)
    data[i] = std::pow(data[i], 1.0f / gamma);
}

void
removeAlpha(float *data, size_t size)
{
  assert(size % 4 == 0);

  for (auto current = 0, insert = 0; current < size;
       current += 4, insert += 3)
  {
    data[insert + 0] = data[current + 0];
    data[insert + 1] = data[current + 1];
    data[insert + 2] = data[current + 2];
  }
}

void
conv8UC4To32FC4(byte *data, size_t size)
{
  auto rgbend = data + size;
  auto rgbp = rgbend - (size / sizeof(float));

  for (auto rgbfp = reinterpret_cast<float *>(data); rgbp != rgbend;
       ++rgbp, ++rgbfp)
  {
    *rgbfp = static_cast<float>(*rgbp) / 255.0f;
  }
}

void
diff(byte *i1_, byte *i2_, size_t size, int th = 5)
{
  auto i1 = reinterpret_cast<uint8_t *>(i1_);
  auto i2 = reinterpret_cast<uint8_t *>(i2_);

  for (auto i = 0; i < size; ++i)
  {
    if (std::abs(int(i1[i]) - int(i2[i])) < th)
      i1[i] = 255;
  }
}

void
conv8UC4To32FC1(byte *data, size_t size)
{
  auto rgbend = data + size;
  auto rgbp = rgbend - (size / sizeof(float));

  for (auto rgbfp = reinterpret_cast<float *>(data); rgbp != rgbend;
       rgbp += 4, ++rgbfp)
  {
    auto b = static_cast<float>(rgbp[0]) / 255.0f;
    auto g = static_cast<float>(rgbp[1]) / 255.0f;
    auto r = static_cast<float>(rgbp[2]) / 255.0f;

    *rgbfp = (b + g + r) / 3.0f;
  }
}

void
conv32FC1To8CU1(byte *data, size_t size)
{
  auto fp = reinterpret_cast<float *>(data);

  for (auto i = 0; i < size; ++i, ++fp, ++data)
    *data = static_cast<byte>(*fp * 255.0f);
}

void
rgb_process(libfreenect2::Frame *frame)
{}

void
depth_process(libfreenect2::Frame *frame)
{
  size_t depth_width = 512, depth_height = 424;
  size_t fsize_depth = depth_width * depth_height;
  size_t total_size_depth = fsize_depth * sizeof(float);

  size_t rgb_width = 1920, rgb_height = 1080;
  size_t fsize_rgb = rgb_width * rgb_height;
  size_t total_size_rgb = fsize_rgb * sizeof(float) * 4;

  if (freenect2.enumerateDevices() == 0)
  {
    fmt::print("No devices connected\n");
    exit(-1);
  }

  auto imgraw_depth_ = std::make_unique<byte[]>(total_size_depth);
  auto imgraw_depth = std::make_unique<byte[]>(total_size_depth);

  fmt::print("Connecting to the device with serial: {}\n", serial);

  auto pipeline = new libfreenect2::OpenGLPacketPipeline;
  dev = freenect2.openDevice(serial, pipeline);

  dev->setColorFrameListener(&listener);
  dev->setIrAndDepthFrameListener(&listener);

  if (fno == 5) fno = 2;

  fmt::print("Connecting to the device\n"
             "Device serial number	: {}\n"
             "Device firmware	: {}\n",
             dev->getSerialNumber(),
             dev->getFirmwareVersion());
}
constexpr auto bfm_ctype = CV_8UC1;

int
main(int argc, char **argv)
{
  continue_flag.test_and_set();
  if (signal(SIGINT, sigint_handler) == SIG_ERR)
  {
    fmt::print("Failed to register signal handler.\n");
    exit(-2);
  }
  libfreenectInit();

  size_t depth_width = 512, depth_height = 424;
  size_t rgb_width = 1920, rgb_height = 1080;

=======
  if (fread(imgraw_rgb_base_.get(), 4, rgb_width * rgb_height, f) !=
      rgb_width * rgb_height)
    return -2;
  fclose(f);

  conv32FC1To8CU1(imgraw_depth_base_.get(), depth_height * depth_width);
  conv32FC1To8CU1(imgraw_depth_.get(), depth_height * depth_width);
  // removeAlpha(reinterpret_cast<float *>(imgraw_rgb_.get()),
  //          rgb_width * rgb_height * sizeof(float));

>>>>>>> origin/3dview
  cv::SimpleBlobDetector::Params params;
  params.filterByArea = true;
  params.minArea = 5;
  params.maxArea = std::numeric_limits<float>::infinity();

  cv::Ptr<cv::SimpleBlobDetector> det =
    cv::SimpleBlobDetector::create(params);

  int c = 0;
  float g = 0.5f;

  int connectivity = 4, itype = CV_16U, ccltype = cv::CCL_WU;
  int lowerb = 5, higherb = 255;
  int lowerb2 = 20, higherb2 = 240;
  int area = 0;
  bool clr = false;
  int p = 0;
  cv::namedWindow(wndname3, cv::WINDOW_AUTOSIZE);
  auto depth_image_g =
    cv::Mat(depth_height, depth_width, CV_8UC1, imgraw_depth.get());
  bbox best_bbox_g;
  shared_t shared{ std::mutex(), depth_image_g, best_bbox_g };
  cv::setMouseCallback(wndname3, mouse_event_handler, &shared);
  //cv::namedWindow(wndname, cv::WINDOW_AUTOSIZE);
  cv::createTrackbar("lowerb", wndname3, &lowerb, 255);
  cv::setTrackbarPos("lowerb", wndname3, 5);
  //cv::createTrackbar("higherb", wndname, &higherb, 255);
  //cv::createTrackbar("lowerb2", wndname, &lowerb2, 255);
  //cv::createTrackbar("higherb2", wndname, &higherb2, 255);
  //cv::createTrackbar("print", wndname, &p, 1);
  while (c != 'q')
  {
    if (!listener.waitForNewFrame(frames, 10 * 1000))
    {
      fmt::print("TIMEDOUT !\n");
      exit(-1);
    }

    libfreenect2::Frame *rgb = frames[libfreenect2::Frame::Color];
    libfreenect2::Frame *ir = frames[libfreenect2::Frame::Ir];
    libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];

    rgb_process(rgb);

    // gamma(imgf_depth, fsize_depth, g);
    diff(imgraw_depth.get(), imgraw_depth_base.get(), fsize_depth);

    auto image_depth_base =
      cv::Mat(depth->height, depth->width, CV_8UC1, depth->data);
    auto image_depth =
      cv::Mat(depth_height, depth_width, CV_8UC1, imgraw_depth.get());
    auto image_rgb =
      cv::Mat(rgb_height, rgb_width, CV_8UC4, imgraw_rgb.get());
    auto image_rgb_base =
      cv::Mat(rgb_height, rgb_width, CV_8UC4, imgraw_rgb_base.get());
    auto image_th = cv::Mat(depth_height, depth_width, CV_8UC1);

    cv::Mat image_depth_th, image_depth_filtered;
    //cv::threshold( image_depth, image_depth_th, lowerb, higherb, cv::THRESH_BINARY_INV);

    auto image_depth_blurred = image_depth.clone();
    postprocessing::blur(image_depth_blurred, (lowerb*2)+1);
    cv::inRange(image_depth_blurred, lowerb2, higherb2, image_depth_filtered);

    cv::Mat labels, stats, centroids;
    cv::connectedComponentsWithStats(
      image_depth_filtered, labels, stats, centroids);

    clr = !clr;

    bbox best_bbox = {};

    int barea = 0;
    for (int i = 0; i < stats.rows; ++i)
    {
      int x = stats.at<int>({ 0, i });
      int y = stats.at<int>({ 1, i });
      int w = stats.at<int>({ 2, i });
      int h = stats.at<int>({ 3, i });
      int area = stats.at<int>({ 4, i });

      if (area >= depth_width * depth_height / 2)
        continue;

      if (area > best_bbox.area)
      {
        best_bbox.area = area;
        best_bbox.x = x;
        best_bbox.y = y;
        best_bbox.w = w;
        best_bbox.h = h;
      }
    }

    // fmt::print("Area: {}\n", best_bbox.area);
    cv::Scalar color(clr ? 255 : 0, 0, 0);
    cv::Rect rect(best_bbox.x, best_bbox.y, best_bbox.w, best_bbox.h);
    cv::rectangle(image_depth, rect, color, 3);
    // std::vector<cv::KeyPoint> kp;
    // det->detect(image_depth, kp);

    // for (auto &k : kp)
    //  fmt::print("{} ", k.pt);
    // puts("");

    // cv::drawKeypoints(image_depth_th, kp, image_depth_th,
    // cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    p = 0;

    //cv::imshow(wndname4, image_depth_orig);
    //cv::imshow(wndname3, image_rgb);
    //cv::imshow(wndname2, image_depth);
    //cv::imshow(wndname, image_depth_base);
    cv::imshow(wndname3, image_depth_blurred);
    cv::imshow(wndname2, image_depth);
    cv::imshow(wndname, image_depth_filtered);
    //cv::imshow(wndname, image_depth_filtered);

    { // Update global state
      std::scoped_lock lck(shared.lock);
      shared.depth_image = image_depth_blurred;
      shared.best_bbox = best_bbox;
    }

    fmt::print("Best bbox: {}\n", best_bbox.area);

    c = cv::waitKey(200);
  }
  dev->stop();
  dev->close();
}
