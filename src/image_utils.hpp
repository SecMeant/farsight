#pragma ocne
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

using byte = unsigned char;
void conv8UC4To32FC4(byte*, size_t);
void conv8UC4To32FC1(byte*, size_t);
void conv32FC1To8CU1(byte*, size_t);

void gamma(float*, size_t, float);
void removeAlpha(float*, size_t);
void diff(byte *, byte*, size_t, char th =10);

