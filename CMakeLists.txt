cmake_minimum_required(VERSION 2.8)
project(farsight)

set(CMAKE_CXX_STANDARD 17)

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

if (BUILD_EXPERIMENTS)
	add_executable(aruco expr/aruco.cc)
	target_link_libraries(aruco ${OpenCV_LIBS} ${LibUSB_LIBRARIES} ${TurboJPEG_LIBRARIES} ${freenect2_LIBRARIES} glfw OpenGL::GL fmt::fmt)
	add_custom_command(TARGET aruco POST_BUILD COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_SOURCE_DIR}/media ${CMAKE_BINARY_DIR}/media)
endif()

add_executable(test ${CXX_SRC})

target_link_libraries(test ${OpenCV_LIBS} ${LibUSB_LIBRARIES} ${TurboJPEG_LIBRARIES} ${freenect2_LIBRARIES} glfw OpenGL::GL fmt::fmt)
set_property(TARGET test PROPERTY CXX_STANDARD 17)

