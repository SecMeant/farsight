#include "image_proc.hpp"
#include <fmt/format.h>

detector::detector(libfreenect2::Freenect2Device::IrCameraParams depth_config, libfreenect2::Freenect2Device::ColorCameraParams rgb_config)
: reg(depth_config, rgb_config)
{
  cv::SimpleBlobDetector::Params params;
  params.filterByArea = true;
  params.minArea = 5;
  params.maxArea = std::numeric_limits<float>::infinity();

  det = cv::SimpleBlobDetector::create(params);
  configScreen =cv::Mat::zeros(cv::Size(depth_width*2 +10, depth_height*2+10), CV_8UC1);
}

bbox detector::detect(int kinectID,const byte *frame_object, size_t size, cv::Mat &image_depth)
{
  int lowerb = 0, higherb = 255;
  int lowerb2 = 20, higherb2 = 240;
  bool clr = false;

  std::array<byte, depth_width*depth_height> frame_depth_;
  std::copy(frame_object,
            frame_object + depth_width*depth_height,
            frame_depth_.data());

  diff(frame_depth_.data(), sceneConfiguration[kinectID].img_base.ptr(), size);
  auto image_depth_ =
    cv::Mat(depth_height, depth_width, CV_8UC1, frame_depth_.data());
  cv::Mat image_depth_th, image_depth_filtered;
  cv::threshold(image_depth_, image_depth_th, lowerb, higherb, cv::THRESH_BINARY_INV);
  cv::inRange(image_depth_, lowerb2, higherb2, image_depth_filtered);
  
  cv::Mat labels, stats, centroids;
  cv::connectedComponentsWithStats(image_depth_filtered, labels, stats, centroids);
  
  clr = !clr;
  bbox best_bbox;

  for (int i = 0; i < stats.rows; ++i)
  {
    int x = stats.at<int>({0,i});
    int y = stats.at<int>({1,i});
    int w = stats.at<int>({2,i});
    int h = stats.at<int>({3,i});
    int area = stats.at<int>({4,i});
  
    if (area >= depth_width * depth_height / 2)
      continue;
  
    if (area > best_bbox.area) {
      best_bbox.area = area;
      best_bbox.x = x;
      best_bbox.y = y;
      best_bbox.w = w;
      best_bbox.h = h;
    }
  }
  
  cv::Scalar color(clr ? 255 : 0, 0, 0);
  cv::Rect rect(best_bbox.x,best_bbox.y,best_bbox.w,best_bbox.h);
  cv::rectangle(image_depth, rect, color, 3);
  return best_bbox;
}

bbox detector::transform(const objectConfig &c)
{
    bbox out;
    const auto& in = c.area;
    const auto& org_dep =  c.originalObjectFrame;
    float rgb_x, rgb_y;

    int pos = in.y*depth_width + in.x;
    reg.apply(in.x, in.y, org_dep[pos], rgb_x, rgb_y);
    out.x = rgb_x;
    out.y = rgb_y;

    pos = (in.y+in.h)*depth_width + (in.x + in.w);
    reg.apply(in.x+in.w, in.y+in.h, org_dep[pos], rgb_x, rgb_y);
    out.w = rgb_x - out.x;
    out.h = rgb_y - out.y;
    return out;
}

void detector::configure(int kinectID,const cv::Mat &img, cv::Mat &rgb,const bbox &sizes, const depth_t &dep)
{
    auto & c = sceneConfiguration[kinectID];
    img.copyTo(c.img_object);
    rgb.copyTo(c.img_rgb);
    c.area = sizes;
    c.rgb_area = transform(c);
    c.dep = dep;
    c.imObjectSet= true;

    cv::Scalar color(0, 0, 0);
    cv::Rect rect(c.rgb_area.x,c.rgb_area.y,c.rgb_area.w,c.rgb_area.h);
    cv::rectangle(c.img_rgb, rect, color, 3);
    cv::imshow("test_to_be_removed", c.img_rgb);
}

void detector::setBaseImg(int kinectID,const cv::Mat &img)
{
    auto & c = sceneConfiguration[kinectID];
    img.copyTo(c.img_base);
    img.copyTo(c.img_base);
    c.imBaseSet= true;
}

void detector::displayCurrectConfig()
{
    const auto & c1 = sceneConfiguration[0];
    const auto & c2 = sceneConfiguration[1];
    cv::Mat temp; 

    if(c1.imBaseSet == true)
    {
        matRoi =cv::Rect(0, 0, depth_width, depth_height);
        resize(c1.img_base, temp, cv::Size(depth_width, depth_height));
        cv::putText(temp, "c1 base", {50,50},cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar::all(255), 3, 5);
        temp.copyTo(configScreen(matRoi));
    }
    
    if(c1.imObjectSet== true)
    {
        matRoi =cv::Rect(depth_width, 0, depth_width,depth_height);
        resize(c1.img_object, temp, cv::Size(depth_width,depth_height));
        cv::putText(temp, fmt::format("object {}x{} pixels ", c1.area.w, c1.area.h), {depth_width/10,50},cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar::all(255), 3, 5);
        cv::putText(temp, fmt::format("object depth {} cm ", c1.dep.depth), {depth_width/10,100},cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar::all(255), 3, 5);
        temp.copyTo(configScreen(matRoi));
    }

    if(c2.imBaseSet == true)
    {
        matRoi =cv::Rect(0, depth_height, depth_width, depth_height);
        resize(c2.img_base, temp, cv::Size(depth_width, depth_height));
        cv::putText(temp, "c2 base", {50,depth_height/10},cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar::all(255), 3, 5);
        temp.copyTo(configScreen(matRoi));
    }
    
    if(c2.imObjectSet== true)
    {
        matRoi =cv::Rect(depth_width, depth_height, depth_width,depth_height);
        resize(c2.img_object, temp, cv::Size(depth_width,depth_height));
        cv::putText(temp, fmt::format("object {}x{} pixels ", c2.area.w, c2.area.h), {depth_width/10,depth_height/10},cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar::all(255), 3, 5);
        cv::putText(temp, fmt::format("object depth {} cm ", c2.dep.depth), {depth_width/10,100},cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar::all(255), 3, 5);
        temp.copyTo(configScreen(matRoi));
    }
    cv::imshow("Dupa", configScreen); 
}

void detector::saveOriginalFrameObject(int kinectID, const libfreenect2::Frame *frame)
{
    auto & c = sceneConfiguration[kinectID];
    std::copy(frame->data,
        frame->data + detector::depth_total_size,
        c.originalObjectFrame.get());
}

void detector::meassure(int kinectID, const bbox & o_area, int depth)
{
   auto &c = sceneConfiguration[kinectID];
   
   // calc alfa, we can use tanges relation her x/h
   double alpha_w_ref = atan(((double(c.hrea.w))/c.dep.depth);
   double alpha_h_ref = atan((double(c.area.h))/c.dep.depth);
   
   double alpha_w = atan((double(o_area.w))/depth);
   double alpha_h = atan((double(o_area.h))/depth);
  
   double object_w = (alpha_w/alpha_w_ref)*cubeWidth;
   double object_h = (alpha_h/alpha_h_ref)*cubeWidth;
   fmt::print("Object size: {} {}\n", object_w, object_h);

}

bool detector::isFullyConfigured()
{
    const auto & c1 = sceneConfiguration[0];
    const auto & c2 = sceneConfiguration[1];

    return c1.imBaseSet && c1.imObjectSet && c2.imBaseSet && c2.imObjectSet;
}
