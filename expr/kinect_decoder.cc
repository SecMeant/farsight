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

constexpr int ir_max_size = 512*424; 
constexpr int rgb_max_size = 1920*1080; 
constexpr int ir_width = 512; 
constexpr int ir_height = 424; 
constexpr int rgb_width = 1920;
constexpr int rgb_height = 1080; 

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
   
   cv::Mat im_rgb(rgb_height, rgb_width, CV_8UC4);

   for(int r =0; r< rgb_height; r++)
   {
    for(int c=0; c< rgb_width; c++)
    {
     auto d = data->rgb.image(r,c);
     im_rgb.at<uint32_t>(r,c) = d;
    }
   }

   cv::Mat im_ir(ir_height, ir_width, CV_8UC1);
   pos = 0;

   float *arr = new float[ir_max_size];
   for(int r =0; r< ir_height; r++)
   {
    for(int c=0; c< ir_width; c++)
    {
        pos = r*ir_width+ c;
        arr[pos] = data->ir.image(r,c);
    }
   }

   std::sort(arr, arr+ir_max_size); 
   int percent_90 = ir_max_size*0.99;
   int val_perc90 = arr[percent_90];
   fprintf(stderr,"%i\n", val_perc90);
   for(int r =0; r< ir_height; r++)
   {
    for(int c=0; c< ir_width; c++)
    {
        im_ir.at<uint8_t>(r,c) = 255*(data->ir.image(r,c)/val_perc90);
    }
   }
    
   

   std::string f_name = fs::path(file_path).filename();
   
   std::string p_path = fs::path(file_path).parent_path();

   std::string dir_rgb = fmt::format("{}/rgb",p_path, data->serialNumber);
   std::string dir_ir = fmt::format("{}/ir", p_path, data->serialNumber);

   if(!fs::exists(dir_rgb))
       fs::create_directory(fmt::format("{}",dir_rgb));

   if(!fs::exists(dir_ir))
       fs::create_directory(fmt::format("{}",dir_ir));

   cv::flip(im_rgb, im_rgb,1);
   cv::flip(im_ir, im_ir,1);

   cv::imwrite(fmt::format("{}/{}.jpg", dir_rgb, f_name), im_rgb);
   cv::imwrite(fmt::format("{}/{}.jpg", dir_ir, f_name), im_ir);

   delete [] arr;
   delete data;
}
