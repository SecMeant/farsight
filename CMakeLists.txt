cmake_minimum_required(VERSION 2.8)
project(farsight)

option(BUILD_EXPERIMENTS "Build exepriments" OFF)

LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)

set(OpenGL_GL_PREFERENCE "GLVND")

find_package(OpenCV REQUIRED)
find_package(LibUSB REQUIRED)
find_package(TurboJPEG REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(freenect2 REQUIRED)
find_package(fmt REQUIRED)

file(GLOB CXX_SRC src/*.cc src/*.cpp)
set(CMAKE_CXX_STANDARD 17)
include_directories(inc/)

if (BUILD_EXPERIMENTS)
	add_executable(aruco expr/aruco.cc)
        add_executable(kinect_decoder expr/kinect_decoder.cc)
	add_executable(aruco_dump expr/aruco_dump.cc src/kinect_manager.cpp)
	add_executable(charuco expr/charuco.cc src/kinect_manager.cpp)
        target_include_directories(aruco_dump PUBLIC src)
        target_include_directories(charuco PUBLIC src)
        target_include_directories(kinect_decoder PUBLIC src)
	target_link_libraries(kinect_decoder ${OpenCV_LIBS} ${LibUSB_LIBRARIES} ${TurboJPEG_LIBRARIES} ${freenect2_LIBRARIES} glfw OpenGL::GL fmt::fmt stdc++fs)
	target_link_libraries(aruco ${OpenCV_LIBS} ${LibUSB_LIBRARIES} ${TurboJPEG_LIBRARIES} ${freenect2_LIBRARIES} glfw OpenGL::GL fmt::fmt)
	target_link_libraries(aruco_dump ${OpenCV_LIBS} ${LibUSB_LIBRARIES} ${TurboJPEG_LIBRARIES} ${freenect2_LIBRARIES} glfw OpenGL::GL fmt::fmt)
	target_link_libraries(charuco ${OpenCV_LIBS} ${LibUSB_LIBRARIES} ${TurboJPEG_LIBRARIES} ${freenect2_LIBRARIES} glfw OpenGL::GL fmt::fmt)
	add_custom_command(TARGET aruco POST_BUILD COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_SOURCE_DIR}/media ${CMAKE_BINARY_DIR}/media)
endif()

add_executable(test ${CXX_SRC})

target_link_libraries(test ${OpenCV_LIBS} ${LibUSB_LIBRARIES} ${TurboJPEG_LIBRARIES} ${freenect2_LIBRARIES} glfw OpenGL::GL glut GLU fmt::fmt)
set_property(TARGET test PROPERTY CXX_STANDARD 17)

