cmake_minimum_required(VERSION 2.8)
project(3dview)

find_package(OpenCV REQUIRED)

set(CXX_SOURCES src/main.cc)
add_executable(3dview ${CXX_SOURCES})
target_link_libraries(3dview glut GL GLU ${OpenCV_LIBS} fmt)
set_property(TARGET 3dview PROPERTY CXX_STANDARD 17)
