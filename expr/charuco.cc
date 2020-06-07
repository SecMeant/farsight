#include <opencv2/aruco/charuco.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <fmt/format.h>
#include "kinect_manager.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace {
const char* about = "A tutorial code on charuco board creation and detection of charuco board with and without camera caliberation";
const char* keys = "{c        |       | Put value of c=1 to create charuco board;\nc=2 to detect charuco board without camera calibration;\nc=3 to detect charuco board with camera calibration and Pose Estimation}";
}
void createBoard();
void detectCharucoBoardWithCalibrationPose();
void detectCharucoBoardWithoutCalibration();
static bool readCameraParameters(std::string filename, cv::Mat& camMatrix, cv::Mat& distCoeffs)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened())
        return false;
    fs["camera_matrix"] >> camMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    return true;
}
void createBoard()
{
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::Ptr<cv::aruco::CharucoBoard> board = cv::aruco::CharucoBoard::create(5, 7, 0.04f, 0.02f, dictionary);
    cv::Mat boardImage;
    board->draw(cv::Size(600, 500), boardImage, 10, 1);
    cv::imwrite("BoardImage.jpg", boardImage);
}

static kinect k_dev(0);
void detectCharucoBoardWithCalibrationPose()
{
    cv::Mat cameraMatrix(3,3, CV_32FC1), distCoeffs(1, 6, CV_32FC1);
    std::string filename = "calib.txt";
    auto colorParams = k_dev.getColorParams();
    cameraMatrix.at<float>(0,0) = colorParams.fx;
    cameraMatrix.at<float>(1,1) = colorParams.fy;
    cameraMatrix.at<float>(0,2) = colorParams.cx;
    cameraMatrix.at<float>(0,0) = colorParams.cy;
    bool readOk = readCameraParameters(filename, cameraMatrix, distCoeffs);
        cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
        cv::Ptr<cv::aruco::CharucoBoard> board = cv::aruco::CharucoBoard::create(5, 7, 0.04f, 0.02f, dictionary);
        cv::Ptr<cv::aruco::DetectorParameters> params = cv::aruco::DetectorParameters::create();
        while (true) {
            k_dev.waitForFrames(10);
            libfreenect2::Frame *rgb = k_dev.frames[libfreenect2::Frame::Color];
            libfreenect2::Frame *ir = k_dev.frames[libfreenect2::Frame::Ir];
            libfreenect2::Frame *depth = k_dev.frames[libfreenect2::Frame::Depth];
            cv::Mat imageCopy;
            auto image =
              cv::Mat(rgb->height, rgb->width, CV_8UC4, rgb->data);
            cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
            image.copyTo(imageCopy);
            std::vector<int> markerIds;
            std::vector<std::vector<cv::Point2f> > markerCorners;
            cv::aruco::detectMarkers(image, board->dictionary, markerCorners, markerIds, params);
            // if at least one marker detected
            if (markerIds.size() > 0) {
                cv::aruco::drawDetectedMarkers(imageCopy, markerCorners, markerIds);
                std::vector<cv::Point2f> charucoCorners;
                std::vector<int> charucoIds;
                cv::aruco::interpolateCornersCharuco(markerCorners, markerIds, image, board, charucoCorners, charucoIds, cameraMatrix, distCoeffs);
                // if at least one charuco corner detected
                if (charucoIds.size() > 0) {
                    cv::Scalar color = cv::Scalar(255, 0, 0);
                    cv::aruco::drawDetectedCornersCharuco(imageCopy, charucoCorners, charucoIds, color);
                    cv::Vec3d rvec, tvec;
                    // cv::aruco::estimatePoseCharucoBoard(charucoCorners, charucoIds, board, cameraMatrix, distCoeffs, rvec, tvec);
                    bool valid = cv::aruco::estimatePoseCharucoBoard(charucoCorners, charucoIds, board, cameraMatrix, distCoeffs, rvec, tvec);
                    // if charuco pose is valid
                    if (valid)
                        cv::aruco::drawAxis(imageCopy, cameraMatrix, distCoeffs, rvec, tvec, 0.1f);
                }
            }
            k_dev.releaseFrames();
            cv::imshow("out", imageCopy);
            char key = (char)cv::waitKey(30);
            if (key == 27)
                break;
        }
}
void detectCharucoBoardWithoutCalibration()
{
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::Ptr<cv::aruco::CharucoBoard> board = cv::aruco::CharucoBoard::create(5, 7, 0.04f, 0.02f, dictionary);
    cv::Ptr<cv::aruco::DetectorParameters> params = cv::aruco::DetectorParameters::create();
    params->cornerRefinementMethod = cv::aruco::CORNER_REFINE_NONE;
    while (true) {
        cv::Mat imageCopy;
        k_dev.waitForFrames(10);
        libfreenect2::Frame *rgb = k_dev.frames[libfreenect2::Frame::Color];
        libfreenect2::Frame *ir = k_dev.frames[libfreenect2::Frame::Ir];
        libfreenect2::Frame *depth = k_dev.frames[libfreenect2::Frame::Depth];
        auto image =
          cv::Mat(rgb->height, rgb->width, CV_8UC4, rgb->data);
        cv::flip(image, image, 1);
        cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
        image.copyTo(imageCopy);
        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f> > markerCorners;
        cv::aruco::detectMarkers(image, board->dictionary, markerCorners, markerIds, params);
        //or
        //cv::aruco::detectMarkers(image, dictionary, markerCorners, markerIds, params);
        // if at least one marker detected
        if (markerIds.size() > 0) {
            cv::aruco::drawDetectedMarkers(imageCopy, markerCorners, markerIds);
            std::vector<cv::Point2f> charucoCorners;
            std::vector<int> charucoIds;
            cv::aruco::interpolateCornersCharuco(markerCorners, markerIds, image, board, charucoCorners, charucoIds);
            // if at least one charuco corner detected
            if (charucoIds.size() > 0)
                cv::aruco::drawDetectedCornersCharuco(imageCopy, charucoCorners, charucoIds, cv::Scalar(255, 0, 0));
        }
        k_dev.releaseFrames();
        cv::imshow("out", imageCopy);
        char key = (char)cv::waitKey(30);
        if (key == 27)
            break;
    }
}

int main(int argc, char* argv[])
{
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about(about);
    if (argc < 2) {
        parser.printMessage();
        return 0;
    }
    int choose = parser.get<int>("c");
    switch (choose) {
    case 1:
        createBoard();
        std::cout << "An image named BoardImg.jpg is generated in folder containing this file" << std::endl;
        break;
    case 2:
        detectCharucoBoardWithoutCalibration();
        break;
    case 3:
        detectCharucoBoardWithCalibrationPose();
        break;
    default:
        break;
    }
    return 0;
}
