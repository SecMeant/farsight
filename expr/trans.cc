struct kincet_color
{
    float fx=1081.37;
    float fy=1081.37;
    float cx=959.5;
    float cy=539.5;
    float shift_d=863;
    float shift_m=52;
    float mx_x3y0=0.000449294;
    float mx_x0y3=1.91656e-05;
    float mx_x2y1=4.82909e-05;
    float mx_x1y2=0.000353673;
    float mx_x2y0=-2.44043e-05;
    float mx_x0y2=-1.19426e-05;
    float mx_x1y1=0.000988431;
    float mx_x1y0=0.642474;
    float mx_x0y1=0.00500649;
    float mx_x0y0=0.142021;
    float my_x3y0=4.42793e-06;
    float my_x0y3=0.000724863;
    float my_x2y1=0.000398557;
    float my_x1y2=4.90383e-05;
    float my_x2y0=0.000136024;
    float my_x0y2=0.00107291;
    float my_x1y1=-1.75465e-05;
    float my_x1y0=-0.00554263;
    float my_x0y1=0.641807;
    float my_x0y0=0.0180811;
}color;

struct kincet_depth
{
    float fx=365.481;
    float fy=365.481;
    float cx=257.346;
    float cy=210.347;
    float k1=0.089026;
    float k2=-0.271706;
    float k3=0.0982151;
    float p1=0;
    float p2=365.481;
}depth;

float depth_q = 0.01;
float color_q = 0.002199;

struct bbox
{
  int x, y, w, h, area;
};

void distort(int mx, int my, float& x, float& y)
{
  // see http://en.wikipedia.org/wiki/Distortion_(optics) for description
  float dx = ((float)mx - depth.cx) / depth.fx;
  float dy = ((float)my - depth.cy) / depth.fy;
  float dx2 = dx * dx;
  float dy2 = dy * dy;
  float r2 = dx2 + dy2;
  float dxdy2 = 2 * dx * dy;
  float kr = 1 + ((depth.k3 * r2 + depth.k2) * r2 + depth.k1) * r2;
  x = depth.fx * (dx * kr + depth.p2 * (r2 + 2 * dx2) + depth.p1 * dxdy2) + depth.cx;
  y = depth.fy * (dy * kr + depth.p1 * (r2 + 2 * dy2) + depth.p2 * dxdy2) + depth.cy;
}

void depth_to_color(float mx, float my, float& rx, float& ry) 
{
  mx = (mx - depth.cx) * depth_q;
  my = (my - depth.cy) * depth_q;

  float wx =
    (mx * mx * mx * color.mx_x3y0) + (my * my * my * color.mx_x0y3) +
    (mx * mx * my * color.mx_x2y1) + (my * my * mx * color.mx_x1y2) +
    (mx * mx * color.mx_x2y0) + (my * my * color.mx_x0y2) + (mx * my * color.mx_x1y1) +
    (mx * color.mx_x1y0) + (my * color.mx_x0y1) + (color.mx_x0y0);

  float wy =
    (mx * mx * mx * color.my_x3y0) + (my * my * my * color.my_x0y3) +
    (mx * mx * my * color.my_x2y1) + (my * my * mx * color.my_x1y2) +
    (mx * mx * color.my_x2y0) + (my * my * color.my_x0y2) + (mx * my * color.my_x1y1) +
    (mx * color.my_x1y0) + (my * color.my_x0y1) + (color.my_x0y0);

  rx = (wx / (color.fx * color_q)) - (color.shift_m / color.shift_d);
  ry = (wy / color_q) + color.cy;
}
 
