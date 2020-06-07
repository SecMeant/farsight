#include <libfreenect2/frame_listener.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/photo.hpp>
#include<string>
#include<fmt/format.h>
#include <filesystem>
namespace fs = std::filesystem;

template<typename T, std::size_t width, std::size_t height>
struct KinectFrame
{
    cv::Matx<T, height, width> image;
    std::uint32_t timestamp;
    std::uint32_t sequence;
    float exposure;
    float gain;
    float gamma;
    uint32_t status;        
    KinectFrame(const libfreenect2::Frame& frame)
        : image(assert(frame.width == width), assert(frame.height == height), (T*)frame.data)
        , timestamp(frame.timestamp)
        , sequence(frame.sequence)
        , exposure(frame.exposure)
        , gain(frame.gain)
        , gamma(frame.gamma)
        , status(frame.status)
    {}        
    KinectFrame()
        : image()
        , timestamp(0)
        , sequence(0)
        , exposure(0.0f)
        , gain(0.0f)
        , gamma(0.0f)
        , status(-1)
    {}        
    operator cv::Mat()
    {
        cv::Mat ret(image, false);           
        if constexpr (width == 1920 && height == 1080)
        {
            ret.flags &= ~cv::Mat::TYPE_MASK;
            ret.flags |= CV_MAT_TYPE(CV_8UC4);
        }
        else
        {
            ret.flags &= ~cv::Mat::TYPE_MASK;
            ret.flags |= CV_MAT_TYPE(CV_32FC1);
        }
        return ret;
    }
};

struct KinectRawPicture
{
    char serialNumber[32];
    std::chrono::system_clock::time_point timestamp;
    KinectFrame<float, 512, 424> ir;
    KinectFrame<float, 512, 424> depth;
    KinectFrame<std::uint32_t, 1920, 1080> rgb;
};

int main(int argc, char **argv)
{
   if(argc != 2)
       return -1;
   const char* file_path = argv[1]; 
   FILE *f = fopen(file_path, "rb");
   if(f == nullptr)
       return -2;

   auto *data = new KinectRawPicture;
   size_t size = sizeof(KinectRawPicture);
   size_t pos = 0;
   while(pos < size){
       size_t r_size = fread(data, 1, size, f);
       if(r_size == -1)
           return -1;
       pos+=r_size;
   }
   fclose(f);
   
   cv::Mat im_rgb(1080, 1920, CV_8UC4);

   for(int r =0; r< 1080; r++)
   {
    for(int c=0; c< 1920; c++)
    {
     auto d = data->rgb.image(r,c);
     im_rgb.at<uint32_t>(r,c) = d;
    }
   }

   cv::Mat im_ir(424, 512, CV_32FC1);
   for(int r =0; r< 424; r++)
   {
    for(int c=0; c< 512; c++)
    {
     auto d = data->ir.image(r,c);
     im_ir.at<float>(r,c) = d;
    }
   }

   std::string dir_rgb = fmt::format("{}_rgb", data->serialNumber);
   std::string dir_ir = fmt::format("{}_ir", data->serialNumber);

   std::string f_name = fs::path(file_path).filename();

   if(!fs::exists(dir_rgb))
       fs::create_directory(fmt::format("{}",dir_rgb));

   if(!fs::exists(dir_ir))
       fs::create_directory(fmt::format("{}",dir_ir));

   cv::flip(im_rgb, im_rgb,1);
   cv::flip(im_ir, im_ir,1);

   cv::imwrite(fmt::format("{}/{}.jpg", dir_rgb, f_name), im_rgb);
   cv::imwrite(fmt::format("{}/{}.jpg", dir_ir, f_name), im_ir);
   return data->serialNumber[0];
}
